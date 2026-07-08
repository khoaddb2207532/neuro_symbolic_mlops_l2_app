"""Pipeline huấn luyện GFlowNet để chọn tập luật con tối ưu.

Refactor: tách vòng lặp train thành 3 mối quan tâm độc lập, để dễ đọc/test
và để phần "train step thuần" match 1-1 với baseline trong intro_discrete.ipynb:

  1. _train_step        — đúng 5 dòng lõi của torchgfn (sample -> loss -> step).
  2. _EliteTracker       — theo dõi rule-set có log-reward cao nhất TỪNG THẤY,
                           độc lập với trọng số model (đây là phần "tìm kiếm
                           tổ hợp tốt nhất", không phải density estimation).
  3. _CheckpointTracker  — EMA hoá log-reward validation để feed scheduler +
                           lưu/khôi phục checkpoint model tốt nhất theo EMA.

_train_gflownet giờ chỉ còn orchestrate 3 phần trên theo đúng thứ tự cũ,
không thay đổi bất kỳ hành vi/số liệu nào so với bản gốc.
"""
import abc
import os
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

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

from dvclive import Live


logger = get_logger(__name__)


class _EliteTracker:
    """Giữ lại rule-set tốt nhất TỪNG THẤY qua mọi lần validate, bất kể model
    hiện tại có tốt hơn hay không. Đây là phần biến GFlowNet thành một công
    cụ tìm kiếm tổ hợp (elitist), tách biệt khỏi việc train sampler."""

    def __init__(self) -> None:
        self.best_log_reward = float("-inf")
        self.best_selected: List[Rule] = []

    def update(self, val_trajectories: list, valid_rules: List[Rule]) -> None:
        for vt in val_trajectories:
            r = vt.log_rewards
            idx = r.argmax().item()
            if r[idx].item() > self.best_log_reward:
                self.best_log_reward = r[idx].item()
                mask = vt.terminating_states.tensor[idx].bool().cpu()
                self.best_selected = [valid_rules[i] for i in torch.where(mask)[0].tolist()]


class _CheckpointTracker:
    """EMA-hoá log-reward validation (để scheduler ổn định hơn) và lưu/khôi
    phục checkpoint của model ứng với EMA tốt nhất."""

    def __init__(self, ckpt_path: str, ema_alpha: float = 0.3) -> None:
        self.ckpt_path = ckpt_path
        self.ema_alpha = ema_alpha
        self.ema_val: Optional[float] = None
        self.best_ema = float("-inf")

    def update(
        self,
        avg_val: float,
        gflownet,
        iteration: int,
        n_valid: int,
        max_rules: int,
        early_stop_delta: float,
    ) -> Tuple[float, bool]:
        """Cập nhật EMA, lưu checkpoint nếu cải thiện. Trả về (ema_val, improved)."""
        self.ema_val = (
            avg_val if self.ema_val is None
            else (1 - self.ema_alpha) * self.ema_val + self.ema_alpha * avg_val
        )
        improved = self.ema_val > self.best_ema + early_stop_delta
        if improved:
            self.best_ema = self.ema_val
            state_dict = {k: v.cpu().clone() for k, v in gflownet.state_dict().items()}
            torch.save(
                {
                    "iteration": iteration,
                    "model": state_dict,
                    "best_log_reward": self.best_ema,
                    "n_rules": n_valid,
                    "max_rules": max_rules,
                },
                self.ckpt_path,
            )
        return self.ema_val, improved

    def restore_best(self, gflownet, device: torch.device) -> float:
        """Load checkpoint tốt nhất vào gflownet (in-place). Trả về best_log_reward
        đã lưu (0.0 nếu không có checkpoint nào, giữ đúng hành vi gốc: trong
        trường hợp đó self.best_ema vẫn ở -inf và không được dùng ở nơi khác)."""
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=device)
            gflownet.load_state_dict(ckpt["model"])
            return ckpt["best_log_reward"]
        return self.best_ema


class BaseGFlowNetPipeline(abc.ABC):

    def __init__(self, device: str = "cuda", grad_clip_max_norm: Optional[float] = 5.0):
        """grad_clip_max_norm: ngưỡng clip gradient. Đặt None để TẮT HẲN clipping
        (dùng để kiểm tra giả thuyết grad-clip đang là nút thắt). Nới từ 1.0 (cũ)
        lên 5.0 làm mặc định mới vì action space lớn (n_valid luật) khiến grad
        norm tự nhiên của layer cuối MLP thường > 1 ngay cả khi hướng đúng."""
        self.device = torch.device(device)
        self.grad_clip_max_norm = grad_clip_max_norm if grad_clip_max_norm is not None else float("inf")

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

    def _reward_params(self) -> Dict[str, Any]:
        """Các hyperparameter riêng của reward function (vd w_acc, w_cov...)
        để DVC log lại cùng params huấn luyện chung. Mặc định rỗng; subclass
        override nếu muốn track thêm."""
        return {}

    # ------------------------------------------------------------------
    # 1) Train step thuần — match 1-1 với vòng lặp lõi trong
    #    intro_discrete.ipynb (cell 59/69): sample -> loss -> backward -> step.
    #    Phần warmup + grad clip được giữ lại vì chi phí gần như 0 và cần
    #    thiết cho bài toán này (không phải "trang trí" thêm).
    # ------------------------------------------------------------------
    def _train_step(
        self,
        gflownet,
        optimizer,
        env: RuleSelectionEnv,
        batch_size: int,
        in_warmup: bool,
    ):
        for p in gflownet.pf_pb_parameters():
            p.requires_grad_(not in_warmup)

        trajectories = gflownet.sample_trajectories(env, n=batch_size, save_logprobs=True)
        samples = gflownet.to_training_samples(trajectories)

        optimizer.zero_grad()
        loss = gflownet.loss(env, samples)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            gflownet.parameters(), max_norm=self.grad_clip_max_norm
        )
        optimizer.step()

        return loss, trajectories, grad_norm

    # ------------------------------------------------------------------
    # 2) Validation — sample nhiều lần để ước lượng log-reward ổn định hơn.
    # ------------------------------------------------------------------
    @staticmethod
    def _run_validation(gflownet, env: RuleSelectionEnv, val_samples: int, n_repeats: int = 3):
        all_vt, raw_vals = [], []
        with torch.no_grad():
            for _ in range(n_repeats):
                vt = gflownet.sample_trajectories(env, n=val_samples, save_logprobs=True)
                all_vt.append(vt)
                raw_vals.append(vt.log_rewards.mean().item())
        return all_vt, float(np.mean(raw_vals))

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
        live: Optional["Live"] = None, # type: ignore
    ) -> List[Rule]:
        elite = _EliteTracker()
        ckpt = _CheckpointTracker(os.path.join(output_dir, "gflownet_best.pth"))

        pbar = tqdm(range(num_iterations), desc="GFlowNet (torchgfn)")
        for it in pbar:
            in_warmup = it < logZ_warmup_steps

            loss, trajectories, grad_norm = self._train_step(gflownet, optimizer, env, batch_size, in_warmup)

            avg_log_r = trajectories.log_rewards.mean().item() if hasattr(trajectories, "log_rewards") else 0.0
            logZ_val = gflownet.logZ.item() if loss_type == "tb" else 0.0
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_log_r=f"{avg_log_r:.3f}",
                grad_norm=f"{grad_norm.item():.2f}",
                logZ=f"{logZ_val:.3f}" if loss_type == "tb" else "N/A",
            )

            if live is not None:
                live.log_metric("train/loss", loss.item())
                live.log_metric("train/avg_log_reward", avg_log_r)
                live.log_metric("train/grad_norm", grad_norm.item())
                live.log_metric("train/lr", optimizer.param_groups[0]["lr"])
                if loss_type == "tb":
                    live.log_metric("train/logZ", logZ_val)

            if not in_warmup and (it + 1) % validation_interval == 0:
                all_vt, avg_val = self._run_validation(gflownet, env, val_samples)

                ema_val, _ = ckpt.update(
                    avg_val, gflownet, it + 1, n_valid, max_rules, early_stop_delta
                )
                scheduler.step(ema_val)

                elite.update(all_vt, valid_rules)

                if live is not None:
                    live.log_metric("val/avg_log_reward", avg_val)
                    live.log_metric("val/ema_log_reward", ema_val)
                    live.log_metric("val/best_ema", ckpt.best_ema)
                    live.log_metric("val/best_log_reward_ever", elite.best_log_reward)

                logger.info("Iter %d: ema=%.4f best (%d rules)", it + 1, ema_val, len(elite.best_selected))
                logger.info(
                    "Loss= %.4f : avg_log_r= %.4f : logZ= %.4f : grad_norm= %.4f : lr= %.6g",
                    loss.item(), avg_log_r, logZ_val,
                    grad_norm.item(), optimizer.param_groups[0]["lr"],
                )

            if live is not None:
                live.next_step()

        ckpt.restore_best(gflownet, self.device)

        final_trajs = gflownet.sample_trajectories(env, n=20, save_logprobs=True)
        term_states = final_trajs.terminating_states.tensor.bool().cpu()
        log_rs = final_trajs.log_rewards.cpu()
        best_idx = log_rs.argmax().item()

        reward_module = env.reward_module
        debug_breakdown(elite.best_selected, valid_rules, reward_module, logger, label="best_selected_ever")

        if log_rs[best_idx].item() > elite.best_log_reward:
            # nếu 20 mẫu cuối tình cờ tốt hơn cả lịch sử -> cập nhật
            final_selected = [valid_rules[i] for i in torch.where(term_states[best_idx])[0].tolist()]
        else:
            final_selected = elite.best_selected

        debug_breakdown(final_selected, valid_rules, reward_module, logger, label="final_selected (returned)")

        final_best_log_reward = max(elite.best_log_reward, log_rs[best_idx].item())
        logger.info("Final: %d rules, best=%.4f", len(final_selected), final_best_log_reward)

        if live is not None:
            live.summary["best_log_reward"] = final_best_log_reward
            live.summary["n_rules_selected"] = len(final_selected)
            live.end()

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
        use_dvc: bool = True,
        dvc_dir: Optional[str] = None,
    ) -> List[Rule]:
        """`valid_rules`/`cover`/`correct`/`rule_len` phải đến từ MỘT lần gọi
        duy nhất `RuleValidator.validate_and_build_tensors()` ở ngoài (stage4)
        — pipeline này KHÔNG tự tính lại cover/correct, tránh quét val set
        lần thứ hai (xem README.md, mục "Reward").

        `use_dvc`: bật/tắt tracking bằng DVCLive (loss, avg_log_reward,
        grad_norm, lr, logZ mỗi bước train; avg/ema/best log-reward mỗi lần
        validate; best_log_reward + n_rules_selected làm summary cuối cùng).
        Cần cài `dvclive` (`pip install dvclive`); nếu thiếu, tự động bỏ qua
        và chỉ log qua logger như trước. `dvc_dir` mặc định là
        `<output_dir>/dvclive`.
        """
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

        # Lưu lại NGUYÊN VẸN thứ tự valid_rules SAU permutation + cover/correct/
        # rule_len tương ứng, kèm cấu hình kiến trúc (loss_type/hidden_dim) và
        # trọng số reward — vì `gflownet_best.pth` (lưu ở _CheckpointTracker,
        # xem class đó phía trên) CHỈ chứa state_dict, KHÔNG chứa permutation.
        # Action index i của policy đã train tương ứng với valid_rules[i] SAU
        # permutation, không phải valid_rules gốc trước khi shuffle — nếu thiếu
        # file này, không cách nào ánh xạ đúng lại action -> luật khi nạp lại
        # gflownet_best.pth cho một quá trình khác (vd bước phân tích ranking
        # hoặc Bayesian marginalization ở stage5), vì permutation dùng
        # torch.randperm không được set_seed cố định riêng theo lần gọi.
        rule_order_path = os.path.join(output_dir, "gflownet_rule_order.pkl")
        with open(rule_order_path, "wb") as f:
            pickle.dump(
                {
                    "valid_rules": valid_rules,           # đã permute, khớp index với cover/correct/rule_len bên dưới
                    "cover": cover.cpu(),
                    "correct": correct.cpu(),
                    "rule_len": rule_len.cpu(),
                    "n_valid": n_valid,
                    "max_rules": max_rules,
                    "loss_type": loss_type,
                    "gfnet_hidden_dim": gfnet_hidden_dim,
                    "w_acc": getattr(self, "w_acc", 1.0),
                    "w_cov": getattr(self, "w_cov", 0.5),
                    "w_conflict": getattr(self, "w_conflict", 0.5),
                    "beta": getattr(self, "beta", 3.0),
                },
                f,
            )
        logger.info("Đã lưu rule order + tensor (khớp index với gflownet_best.pth) tại %s", rule_order_path)

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

        live = None
        if use_dvc:
            if Live is None:
                logger.warning(
                    "use_dvc=True nhưng chưa cài dvclive (`pip install dvclive`) — bỏ qua tracking DVC."
                )
            else:
                live = Live(dir=dvc_dir or os.path.join(output_dir, "dvclive"), report="html")
                live.log_params(
                    {
                        "loss_type": loss_type,
                        "n_valid_rules": n_valid,
                        "max_rules": max_rules,
                        "num_iterations": num_iterations,
                        "batch_size": batch_size,
                        "lr": lr,
                        "logZ_lr": logZ_lr,
                        "gfnet_hidden_dim": gfnet_hidden_dim,
                        "validation_interval": validation_interval,
                        "logZ_warmup_steps": logZ_warmup_steps,
                        "val_samples": val_samples,
                        "early_stop_delta": early_stop_delta,
                        "grad_clip_max_norm": self.grad_clip_max_norm,
                        **self._reward_params(),
                    }
                )

        try:
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
                live=live,
            )
        except Exception:
            if live is not None:
                live.end()
            raise


class RuleExtractionPipeline(BaseGFlowNetPipeline):
    """Reward = accuracy + coverage - conflict_redundancy, tính hoàn toàn
    bằng tensor cover/correct/rule_len/targets đã được build sẵn từ bên ngoài.
    Không sklearn, không proxy net trong training loop.

    Đã bỏ w_red (trùng lặp cùng target) và w_comp (số lượng luật) khỏi công
    thức — reward này phục vụ MỤC ĐÍCH DUY NHẤT là chọn luật tốt cho
    regularization ở stage5, không phải để tối ưu tính gọn/dễ đọc của tập
    luật (xem docstring chi tiết trong src/gflownet/reward.py)."""

    def __init__(
        self,
        device: str = "cuda",
        w_acc: float = 1.0,
        w_cov: float = 0.5,
        w_conflict: float = 0.5,
        beta: float = 3.0,
        grad_clip_max_norm : Optional[float] = None, 
    ):
        super().__init__(device, grad_clip_max_norm )
        self.w_acc, self.w_cov, self.w_conflict, self.beta = w_acc, w_cov, w_conflict, beta

    def _reward_params(self) -> Dict[str, Any]:
        return {
            "w_acc": self.w_acc,
            "w_cov": self.w_cov,
            "w_conflict": self.w_conflict,
            "beta": self.beta,
        }

    def _create_reward_function(
        self,
        valid_rules: List[Rule],
        cover: torch.Tensor,
        correct: torch.Tensor,
        rule_len: torch.Tensor,
        max_rules: int,
    ) -> Callable:
        targets = torch.tensor([r.target_class for r in valid_rules], device=cover.device)
        reward_module = RuleSetReward(
            cover=cover,
            correct=correct,
            rule_len=rule_len,
            max_rules=max_rules,
            targets=targets,
            w_acc=self.w_acc,
            w_cov=self.w_cov,
            w_conflict=self.w_conflict,
            beta=self.beta,
        )
        self._last_reward_module = reward_module

        return reward_module