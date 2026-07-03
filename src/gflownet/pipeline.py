"""Pipeline huấn luyện GFlowNet để chọn tập luật con tối ưu."""
import abc
import os
import random
from typing import Callable, List

import numpy as np
import torch
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import DBGFlowNet, FMGFlowNet, TBGFlowNet
from gfn.utils.modules import MLP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

from src.gflownet.env import RuleSelectionEnv
from src.models.proxy_reward import ProxyRewardNet
from src.rules.io import save_rules_excel  # dùng chung, xem src/rules/io.py
from src.rules.penalty import BinaryTransformer
from src.rules.rule_types import Rule, RuleSet
from src.rules.validator import GPUFastRuleValidator
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseGFlowNetPipeline(abc.ABC):
    def __init__(self, min_support: float = 0.01, min_confidence: float = 1.0, device: str = "cuda"):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.device = torch.device(device)

    @abc.abstractmethod
    def _create_reward_function(
        self, train_features, train_labels, val_features, val_labels, valid_rules, proxy_epochs, num_classes
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
        best_selected: List[Rule] = []
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
                raw_vals = []
                with torch.no_grad():
                    for _ in range(3):
                        vt = gflownet.sample_trajectories(env, n=val_samples, save_logprobs=True)
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
                    term = vt.terminating_states.tensor
                    log_r = vt.log_rewards
                    best_idx = log_r.argmax().item()
                    best_mask_tensor = term[best_idx].bool().cpu()
                    best_selected = [valid_rules[i] for i in torch.where(best_mask_tensor)[0].tolist()]
                    logger.info("Iter %d: ema=%.4f best (%d rules)", it + 1, ema_val, len(best_selected))

        if best_state_dict is not None:
            gflownet.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})
        elif os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            gflownet.load_state_dict(ckpt["model"])
            best_log_reward = ckpt["best_log_reward"]

        final_trajs = gflownet.sample_trajectories(env, n=20, save_logprobs=True)
        term_states = final_trajs.terminating_states.tensor.bool().cpu()
        log_rs = final_trajs.log_rewards.cpu()
        best_final = term_states[log_rs.argmax().item()]
        final_indices = torch.where(best_final)[0].tolist()
        final_selected = [valid_rules[i] for i in final_indices] if final_indices else best_selected

        logger.info("Final: %d rules, best_log_reward=%.4f", len(final_selected), best_log_reward)
        return final_selected

    def run(
        self,
        raw_rule_set: RuleSet,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        max_rules: int,
        output_dir: str,
        gfnet_hidden_dim: int = 256,
        num_iterations: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        logZ_lr: float = 1e-2,
        proxy_epochs: int = 5,
        device: str = "cuda",
        validation_interval: int = 100,
        loss_type: str = "tb",
        logZ_warmup_steps: int = 50,
        val_samples: int = 10,
        early_stop_delta: float = 0.001,
    ) -> List[Rule]:
        self.device = torch.device(device)

        validator = GPUFastRuleValidator(self.min_support, self.min_confidence)
        valid_rule_set = validator.validate(raw_rule_set, val_features, val_labels)
        valid_rules = valid_rule_set.rules
        if not valid_rules:
            logger.warning("Không có luật nào hợp lệ sau khi validate.")
            return []

        n_valid = len(valid_rules)
        num_classes = int(torch.unique(train_labels).numel())
        logger.info("Số luật hợp lệ: %d | loss_type: %s", n_valid, loss_type)

        random.shuffle(valid_rules)
        os.makedirs(output_dir, exist_ok=True)
        save_rules_excel(valid_rules, os.path.join(output_dir, "valid_rules.xlsx"))

        reward_fn = self._create_reward_function(
            train_features, train_labels, val_features, val_labels, valid_rules, proxy_epochs, num_classes
        )
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


class ImprovedRuleExtractionPipeline(BaseGFlowNetPipeline):
    """Pipeline V1: reward = accuracy(LR) + coverage + entropy (chậm, chạy trên CPU/sklearn)."""

    def _create_reward_function(self, train_features, train_labels, val_features, val_labels, valid_rules, proxy_epochs, num_classes):
        def reward_fn(states: torch.Tensor) -> torch.Tensor:
            B = states.shape[0] if states.dim() == 2 else 1
            if states.dim() == 1:
                states = states.unsqueeze(0)
            results = torch.zeros(B, device=states.device)

            for b in range(B):
                mask = states[b].bool()
                if mask.sum().item() == 0:
                    continue
                sel_idx = torch.where(mask)[0].tolist()
                subset_rules = [valid_rules[i] for i in sel_idx]
                n_selected = len(subset_rules)

                targets = [r.target_class for r in subset_rules]
                counts = torch.bincount(torch.tensor(targets), minlength=num_classes)
                coverage = (counts > 0).sum().item() / num_classes
                probs = counts.float() / n_selected
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                norm_ent = entropy / (np.log(num_classes) if num_classes > 1 else 1.0)

                transformer = BinaryTransformer()
                train_bin = transformer.transform(train_features, RuleSet(rules=subset_rules)).cpu().numpy()
                val_bin = transformer.transform(val_features, RuleSet(rules=subset_rules)).cpu().numpy()
                clf = OneVsRestClassifier(LogisticRegression(max_iter=100, n_jobs=1))
                clf.fit(train_bin, train_labels.cpu().numpy())
                acc = accuracy_score(val_labels.cpu().numpy(), clf.predict(val_bin))

                size_pen = min(0.05, 0.001 * n_selected)
                results[b] = float(0.5 * acc + 0.3 * coverage + 0.2 * norm_ent - size_pen)

            return results.clamp(min=1e-30)

        return reward_fn


class ImprovedRuleExtractionPipelineV2(BaseGFlowNetPipeline):
    """Pipeline V2: dùng ProxyRewardNet (GPU) pretrain trên slow_reward_fn(sklearn)."""

    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 1.0,
        device: str = "cuda",
        proxy_cache_path: str = None,
        proxy_samples: int = 3000,
        proxy_epochs: int = 30,
    ):
        super().__init__(min_support, min_confidence, device)
        self.proxy_cache_path = proxy_cache_path
        self.proxy_samples = proxy_samples
        self.proxy_epochs = proxy_epochs

    def _create_reward_function(self, train_features, train_labels, val_features, val_labels, valid_rules, proxy_epochs, num_classes):
        n_rules = len(valid_rules)
        device = self.device
        cache_path = self.proxy_cache_path

        def slow_reward_fn(selected_vector: torch.Tensor) -> float:
            rule_mask = selected_vector.bool()
            if rule_mask.sum().item() == 0:
                return 1e-30
            sel_idx = torch.where(rule_mask)[0].tolist()
            subset_rules = [valid_rules[i] for i in sel_idx]
            transformer = BinaryTransformer()
            val_bin = transformer.transform(val_features, RuleSet(rules=subset_rules))
            if val_bin.shape[1] == 0:
                return 1e-30
            clf = OneVsRestClassifier(LogisticRegression(max_iter=100, n_jobs=1))
            clf.fit(val_bin.cpu().numpy(), val_labels.cpu().numpy())
            return float(accuracy_score(val_labels.cpu().numpy(), clf.predict(val_bin.cpu().numpy())))

        proxy_net = ProxyRewardNet(n_rules=n_rules, hidden_dim=128).to(device)

        if cache_path and os.path.exists(cache_path):
            logger.info("Loading ProxyRewardNet từ cache: %s", cache_path)
            proxy_net.load_state_dict(torch.load(cache_path, map_location=device))
        else:
            proxy_net.pretrain(
                true_reward_fn=slow_reward_fn,
                n_rules=n_rules,
                device=device,
                n_samples=self.proxy_samples,
                epochs=self.proxy_epochs,
                lr=1e-3,
            )
            if cache_path:
                torch.save(proxy_net.state_dict(), cache_path)

        proxy_net.eval()
        for p in proxy_net.parameters():
            p.requires_grad_(False)

        def fast_reward_fn(states: torch.Tensor) -> torch.Tensor:
            if states.dim() == 1:
                states = states.unsqueeze(0)
            with torch.no_grad():
                rewards = proxy_net(states.float().to(device)).clamp(min=1e-30, max=1.0)
            return rewards.squeeze(-1)

        return fast_reward_fn
