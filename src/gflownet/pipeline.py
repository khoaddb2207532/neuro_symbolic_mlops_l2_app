"""Pipeline huấn luyện GFlowNet để chọn tập luật con tối ưu."""
import abc
import os
import random
from typing import Callable, List, Optional

import numpy as np
import torch
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import DBGFlowNet, FMGFlowNet, TBGFlowNet
from gfn.utils.modules import MLP
from tqdm import tqdm

from src.gflownet.env import RuleSelectionEnv
from src.gflownet.reward import RuleSetReward
from src.gflownet.evaluation import debug_breakdown
from src.rules.io import save_rules_excel
from src.rules.rule_types import Rule, RuleSet
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseGFlowNetPipeline(abc.ABC):

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

    @abc.abstractmethod
    def _create_reward_function(
        self,
        valid_rules: List[Rule],
        cover: torch.Tensor,
        correct: torch.Tensor,
        rule_len: torch.Tensor,
        max_rules: int,
    ) -> Callable:
        ...

    def _train_gflownet(
        self,
        gflownet,
        optimizer,
        scheduler,
        env: RuleSelectionEnv,
        valid_rules: List[Rule],
        n_valid: int,
        max_rules: int,
        num_iterations: int,
        batch_size: int,
        validation_interval: int,
        logZ_warmup_steps: int,
        val_samples: int,
        early_stop_delta: float,
        loss_type: str,
        output_dir: str,
    ) -> List[Rule]:
        best_log_reward = float("-inf")
        best_log_reward_ever = float("-inf")  
        best_selected_ever: List[Rule] = []     

        best_state_dict = None
        ema_val = None
        ema_alpha = 0.3
        best_ckpt_path = os.path.join(output_dir, "gflownet_best.pth")

        pbar = tqdm(range(num_iterations), desc="GFlowNet (torchgfn)")
        for it in pbar:
            in_warmup = it < logZ_warmup_steps
            for p in gflownet.pf_pb_parameters():
                p.requires_grad_(not in_warmup)

            trajectories = gflownet.sample_trajectories(env, n=batch_size, save_logprobs=True)
            samples = gflownet.to_training_samples(trajectories)

            optimizer.zero_grad()
            loss = gflownet.loss(env, samples)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gflownet.parameters(), max_norm=1.0)
            optimizer.step()

            avg_log_r = trajectories.log_rewards.mean().item() if hasattr(trajectories, "log_rewards") else 0.0
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_log_r=f"{avg_log_r:.3f}",
                logZ=f"{gflownet.logZ.item():.3f}" if loss_type == "tb" else "N/A",
            )

            if not in_warmup and (it + 1) % validation_interval == 0:
                all_vt, raw_vals = [], []
                with torch.no_grad():
                    for _ in range(3):
                        vt = gflownet.sample_trajectories(env, n=val_samples, save_logprobs=True)
                        all_vt.append(vt)
                        raw_vals.append(vt.log_rewards.mean().item())

                avg_val = float(np.mean(raw_vals))
                ema_val = avg_val if ema_val is None else (1 - ema_alpha) * ema_val + ema_alpha * avg_val
                scheduler.step(ema_val)

                if ema_val > best_log_reward + early_stop_delta:
                    best_log_reward = ema_val
                    best_state_dict = {k: v.cpu().clone() for k, v in gflownet.state_dict().items()}
                    torch.save(
                        {
                            "iteration": it + 1,
                            "model": best_state_dict,
                            "best_log_reward": best_log_reward,
                            "n_rules": n_valid,
                            "max_rules": max_rules,
                        },
                        best_ckpt_path,
                    )
                for vt in all_vt:
                    r = vt.log_rewards
                    idx = r.argmax().item()
                    if r[idx].item() > best_log_reward_ever:
                        best_log_reward_ever = r[idx].item()
                        mask = vt.terminating_states.tensor[idx].bool().cpu()
                        best_selected_ever = [valid_rules[i] for i in torch.where(mask)[0].tolist()]
                logger.info("Iter %d: ema=%.4f best (%d rules)", it + 1, ema_val, len(best_selected_ever))

        if best_state_dict is not None:
            gflownet.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})
        elif os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            gflownet.load_state_dict(ckpt["model"])
            best_log_reward = ckpt["best_log_reward"]

        final_trajs = gflownet.sample_trajectories(env, n=20, save_logprobs=True)
        term_states = final_trajs.terminating_states.tensor.bool().cpu()
        log_rs = final_trajs.log_rewards.cpu()
        best_idx = log_rs.argmax().item()

        reward_module = getattr(env.reward_fn, "reward_module", None)
        debug_breakdown(best_selected_ever, valid_rules, reward_module, logger, label="best_selected_ever")


        if log_rs[best_idx].item() > best_log_reward_ever:
            # nếu 20 mẫu cuối tình cờ tốt hơn cả lịch sử -> cập nhật
            final_selected = [valid_rules[i] for i in torch.where(term_states[best_idx])[0].tolist()]
        else:
            final_selected = best_selected_ever

        debug_breakdown(final_selected, valid_rules, reward_module, logger, label="final_selected (returned)")


        logger.info("Final: %d rules, best=%.4f", len(final_selected), max(best_log_reward_ever, log_rs[best_idx].item()))
        return final_selected

    def run(
        self,
        valid_rules: List[Rule],
        cover: torch.Tensor,
        correct: torch.Tensor,
        rule_len: torch.Tensor,
        max_rules: int,
        output_dir: str,
        gfnet_hidden_dim: int = 256,
        num_iterations: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        logZ_lr: float = 1e-2,
        device: str = "cuda",
        validation_interval: int = 100,
        loss_type: str = "tb",
        logZ_warmup_steps: int = 50,
        val_samples: int = 10,
        early_stop_delta: float = 0.001,
    ) -> List[Rule]:
        """`valid_rules`/`cover`/`correct`/`rule_len` phải đến từ MỘT lần gọi
        duy nhất `RuleValidator.validate_and_build_tensors()` ở ngoài (stage4)
        — pipeline này KHÔNG tự tính lại cover/correct, tránh quét val set
        lần thứ hai (xem README.md, mục "Reward")."""
        self.device = torch.device(device)

        if not valid_rules:
            logger.warning("valid_rules rỗng — không có luật nào để GFlowNet chọn.")
            return []

        n_valid = len(valid_rules)
        logger.info("Số luật hợp lệ: %d | loss_type: %s", n_valid, loss_type)

        # Giữ đồng bộ thứ tự giữa valid_rules và các tensor cover/correct/rule_len
        # khi shuffle: shuffle chỉ số rồi hoán vị tensor theo cùng permutation,
        # không random.shuffle(valid_rules) riêng lẻ như trước (sẽ làm lệch hàng).
        perm = torch.randperm(n_valid)
        valid_rules = [valid_rules[i] for i in perm.tolist()]
        cover = cover[perm].to(self.device)
        correct = correct[perm].to(self.device)
        rule_len = rule_len[perm].to(self.device)

        os.makedirs(output_dir, exist_ok=True)
        save_rules_excel(valid_rules, os.path.join(output_dir, "valid_rules.xlsx"))

        reward_fn = self._create_reward_function(valid_rules, cover, correct, rule_len, max_rules)
        env = RuleSelectionEnv(n_valid, max_rules, reward_fn, device=self.device)

        pf_module = MLP(input_dim=env.state_shape[-1], output_dim=env.n_actions, hidden_dim=gfnet_hidden_dim, n_hidden_layers=2)
        pb_module = MLP(input_dim=env.state_shape[-1], output_dim=env.n_actions - 1, hidden_dim=gfnet_hidden_dim, n_hidden_layers=2)

        pf_estimator = DiscretePolicyEstimator(module=pf_module, n_actions=env.n_actions, is_backward=False, preprocessor=env.preprocessor)
        pb_estimator = DiscretePolicyEstimator(module=pb_module, n_actions=env.n_actions, is_backward=True, preprocessor=env.preprocessor)

        if loss_type == "tb":
            gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, init_logZ=0.0)
            optimizer = torch.optim.Adam(
                [
                    {"params": list(gflownet.pf_pb_parameters()), "lr": lr, "weight_decay": 1e-5},
                    {"params": list(gflownet.logz_parameters()), "lr": logZ_lr, "weight_decay": 0.0},
                ]
            )
        elif loss_type == "db":
            logF_module = MLP(input_dim=env.state_shape[-1], output_dim=1, hidden_dim=gfnet_hidden_dim, n_hidden_layers=2)
            logF_estimator = ScalarEstimator(module=logF_module, preprocessor=env.preprocessor)
            gflownet = DBGFlowNet(pf=pf_estimator, pb=pb_estimator, logF=logF_estimator)
            optimizer = torch.optim.Adam(
                [
                    {"params": list(gflownet.pf_pb_parameters()), "lr": lr, "weight_decay": 1e-5},
                    {"params": list(logF_estimator.parameters()), "lr": lr * 2, "weight_decay": 0.0},
                ]
            )
        elif loss_type == "fm":
            gflownet = FMGFlowNet(estimator=pf_estimator)
            optimizer = torch.optim.Adam(gflownet.parameters(), lr=lr, weight_decay=1e-5)
        else:
            raise ValueError(f"loss_type phải là 'tb'/'db'/'fm', nhận '{loss_type}'")

        gflownet.to(self.device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, threshold=1e-4)

        return self._train_gflownet(
            gflownet=gflownet,
            optimizer=optimizer,
            scheduler=scheduler,
            env=env,
            valid_rules=valid_rules,
            n_valid=n_valid,
            max_rules=max_rules,
            num_iterations=num_iterations,
            batch_size=batch_size,
            validation_interval=validation_interval,
            logZ_warmup_steps=logZ_warmup_steps,
            val_samples=val_samples,
            early_stop_delta=early_stop_delta,
            loss_type=loss_type,
            output_dir=output_dir,
        )
    

class RuleExtractionPipeline(BaseGFlowNetPipeline):
    """Reward = accuracy + coverage - redundancy - complexity, tính hoàn toàn
    bằng tensor cover/correct/rule_len đã được build sẵn từ bên ngoài
    (RuleValidator.validate_and_build_tensors). Không sklearn, không proxy
    net trong training loop."""

    def __init__(
        self,
        device: str = "cuda",
        w_acc: float = 1.0,
        w_cov: float = 0.5,
        w_red: float = 0.3,
        w_comp: float = 0.2,
        beta: float = 3.0,
    ):
        super().__init__(device)
        self.w_acc, self.w_cov, self.w_red, self.w_comp, self.beta = w_acc, w_cov, w_red, w_comp, beta

    def _create_reward_function(
        self,
        valid_rules: List[Rule],
        cover: torch.Tensor,
        correct: torch.Tensor,
        rule_len: torch.Tensor,
        max_rules: int,
    ) -> Callable:
        reward_module = RuleSetReward(
            cover=cover,
            correct=correct,
            rule_len=rule_len,
            max_rules=max_rules,
            w_acc=self.w_acc,
            w_cov=self.w_cov,
            w_red=self.w_red,
            w_comp=self.w_comp,
            beta=self.beta,
        )
        self._last_reward_module = reward_module

        def reward_fn(states: torch.Tensor) -> torch.Tensor:
            if states.dim() == 1:
                states = states.unsqueeze(0)
            return reward_module(states.to(self.device))
        
        reward_fn.reward_module = reward_module

        return reward_fn