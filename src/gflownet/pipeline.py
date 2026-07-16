"""Pipeline huấn luyện GFlowNet để chọn tập luật con tối ưu.

Cấu trúc 4 giai đoạn (tối giản, bỏ các cơ chế theo dõi trùng lặp của bản cũ):

  1. Khởi tạo model/optimizer/replay buffer         (`run`, trước khi gọi `_train_gflownet`)
  2. Vòng lặp train off-policy qua replay buffer     (`_train_step`, `gfn.containers.ReplayBuffer`)
  3. Validate định kỳ + early-stop khi mất diversity  (`_evaluate_during_training`,
                                                        `_check_early_stopping`, `_CheckpointTracker`)
  4. Sinh nhiều tập luật ứng viên rồi CHỌN THEO HOLDOUT
     (test set riêng nếu có, không thì cảnh báo dùng tạm val) (`_select_best_on_holdout`)

Đã BỎ so với bản trước: `_EliteTracker` (theo dõi rule-set tốt nhất qua mọi lần
validate — nay thay bằng bước sinh + chọn tường minh ở giai đoạn 4),
`_SamplerCheckpointTracker` + các chỉ số diversity phụ (entropy_norm/top1_share/
calib_corr — chỉ còn giữ `mode_diversity` dùng cho early-stopping).
"""
import abc
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import DBGFlowNet, FMGFlowNet, TBGFlowNet
from gfn.utils.modules import MLP
from tqdm import tqdm

from src.gflownet.env import RuleSelectionEnv
from src.gflownet.reward import RuleSetReward
from src.gflownet.evaluation import debug_breakdown, evaluate_run
from src.rules.io import save_rules_excel
from src.rules.rule_types import Rule, RuleSet
from src.utils.logging_utils import get_logger

from dvclive import Live


logger = get_logger(__name__)


class _CheckpointTracker:
    """Lưu/khôi phục checkpoint model tại điểm val_loss THẤP NHẤT từng thấy
    (không còn EMA hoá — val_loss ở đây đã là trung bình trên `val_samples`
    quỹ đạo mỗi lần validate nên đủ ổn định để so sánh trực tiếp)."""

    def __init__(self, ckpt_path: str) -> None:
        self.ckpt_path = ckpt_path
        self.best_val_loss = float("inf")

    def update(self, val_loss: float, gflownet, iteration: int) -> bool:
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
            state_dict = {k: v.cpu().clone() for k, v in gflownet.state_dict().items()}
            torch.save(
                {"iteration": iteration, "model": state_dict, "best_val_loss": self.best_val_loss},
                self.ckpt_path,
            )
        return improved

    def restore_best(self, gflownet, device: torch.device) -> float:
        """Load checkpoint tốt nhất vào gflownet (in-place). Trả về best_val_loss
        đã lưu (giữ nguyên self.best_val_loss nếu chưa từng lưu checkpoint nào)."""
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=device)
            gflownet.load_state_dict(ckpt["model"])
            return ckpt["best_val_loss"]
        return self.best_val_loss


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
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Callable:
        ...

    def _reward_params(self) -> Dict[str, Any]:
        """Các hyperparameter riêng của reward function (vd alpha, lambda_1...)
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
        replay_buffer: ReplayBuffer,
        batch_size: int,
        in_warmup: bool,
    ):
        for p in gflownet.pf_pb_parameters():
            p.requires_grad_(not in_warmup)

        # Sample quỹ đạo mới -> nạp vào buffer thư viện (`gfn.containers.
        # ReplayBuffer`) -> rút ngẫu nhiên `batch_size` quỹ đạo từ TOÀN BỘ
        # buffer (không nhất thiết là đúng batch vừa thêm) để tính loss.
        trajectories = gflownet.sample_trajectories(env, n=batch_size, save_logprobs=True)
        replay_buffer.add(trajectories)
        batch = replay_buffer.sample(batch_size)
        samples = gflownet.to_training_samples(batch)

        optimizer.zero_grad()
        if self.loss_type == "fm":
            loss = gflownet.loss(env, samples)
        else:
            # recalculate_all_logprobs=True: batch có thể chứa quỹ đạo CŨ lấy từ
            # buffer (off-policy) — log-prob lưu sẵn không còn khớp tham số
            # hiện tại nên phải tính lại để gradient đúng.
            loss = gflownet.loss(env, samples, recalculate_all_logprobs=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            gflownet.parameters(), max_norm=self.grad_clip_max_norm
        )
        optimizer.step()

        return loss, trajectories, grad_norm

    # ------------------------------------------------------------------
    # 2) Validation — 1 batch quỹ đạo mới: val_loss (để scheduler + checkpoint)
    #    + mode_diversity (tỉ lệ tập luật KHÔNG trùng nhau, để phát hiện
    #    mode-collapse và early-stop).
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate_during_training(gflownet, env: RuleSelectionEnv, val_samples: int) -> Dict[str, float]:
        with torch.no_grad():
            vt = gflownet.sample_trajectories(env, n=val_samples, save_logprobs=True)
            val_loss = gflownet.loss(env, gflownet.to_training_samples(vt)).item()

            states = vt.terminating_states.tensor.bool().cpu()
            n_unique = torch.unique(states, dim=0).shape[0]
            mode_diversity = n_unique / states.shape[0]

        return {"val_loss": val_loss, "mode_diversity": mode_diversity}

    @staticmethod
    def _check_early_stopping(
        val_loss_history: List[float], mode_diversity: float,
        patience: int = 10, min_diversity: float = 0.2,
    ) -> bool:
        """Dừng sớm nếu diversity sụp đổ (mode collapse) HOẶC val_loss không
        tạo best mới trong `patience` lần validate liên tiếp."""
        if mode_diversity < min_diversity:
            return True
        if len(val_loss_history) <= patience:
            return False

        best_before_patience_window = min(val_loss_history[:-patience])
        best_in_patience_window = min(val_loss_history[-patience:])
        return best_in_patience_window >= best_before_patience_window

    # ------------------------------------------------------------------
    # 4) Post-training: sinh nhiều tập luật ứng viên rồi chọn tập tốt nhất
    #    theo một RuleSetReward ĐỘC LẬP (nên xây từ tập test/holdout, khác
    #    dữ liệu đã dùng để train reward, tránh chọn theo đúng dữ liệu đã tối ưu).
    # ------------------------------------------------------------------
    @staticmethod
    def _select_best_on_holdout(
        gflownet, env: RuleSelectionEnv, valid_rules: List[Rule],
        num_candidates: int, reward_module_holdout,
    ) -> Tuple[List[Rule], Dict[str, float]]:
        with torch.no_grad():
            trajs = gflownet.sample_trajectories(env, n=num_candidates, save_logprobs=False)
        states = trajs.terminating_states.tensor.bool().cpu()
        unique_states = torch.unique(states, dim=0)

        s = unique_states.float().to(reward_module_holdout.cover.device)
        scores = reward_module_holdout.score(s)
        best_idx = scores.argmax().item()

        best_selected = [valid_rules[i] for i in torch.where(unique_states[best_idx])[0].tolist()]
        report = evaluate_run(best_selected, valid_rules, reward_module_holdout)
        return best_selected, report

    def _train_gflownet(
        self,
        gflownet,
        optimizer,
        scheduler,
        env: RuleSelectionEnv,
        valid_rules: List[Rule],
        num_iterations: int,
        batch_size: int,
        validation_interval: int,
        logZ_warmup_steps: int,
        val_samples: int,
        loss_type: str,
        output_dir: str,
        live: Optional["Live"] = None,  # type: ignore
        replay_capacity: int = 10000,
        num_candidates: int = 100,
        reward_module_holdout=None,
    ) -> List[Rule]:

        self.loss_type = loss_type

        # capacity ở đây là SỐ QUỸ ĐẠO thật (không cần quy đổi qua batch như
        # bản tự viết trước) — mặc định uniform sampling (prioritized_capacity/
        # prioritized_sampling=False), có thể bật lại nếu muốn ưu tiên theo reward.
        replay_buffer = ReplayBuffer(env, capacity=replay_capacity)
        ckpt = _CheckpointTracker(os.path.join(output_dir, "gflownet_best.pth"))
        val_loss_history: List[float] = []

        logger.info("Bắt đầu huấn luyện GFlowNet...")
        pbar = tqdm(range(num_iterations), desc="GFlowNet (torchgfn)")
        for it in pbar:
            in_warmup = it < logZ_warmup_steps

            loss, trajectories, grad_norm = self._train_step(
                gflownet, optimizer, env, replay_buffer, batch_size, in_warmup
            )

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
                val_metrics = self._evaluate_during_training(gflownet, env, val_samples)
                val_loss_history.append(val_metrics["val_loss"])

                scheduler.step(val_metrics["val_loss"])
                ckpt.update(val_metrics["val_loss"], gflownet, it + 1)

                if live is not None:
                    live.log_metric("val/loss", val_metrics["val_loss"])
                    live.log_metric("val/mode_diversity", val_metrics["mode_diversity"])
                    live.log_metric("val/best_loss", ckpt.best_val_loss)

                logger.info(
                    "Epoch %d | Train Loss: %.4f | Val Loss: %.4f | Val Diversity Ratio: %.1f%%",
                    it + 1, loss.item(), val_metrics["val_loss"], val_metrics["mode_diversity"] * 100,
                )

                if self._check_early_stopping(val_loss_history, val_metrics["mode_diversity"]):
                    logger.info("Dừng huấn luyện sớm để bảo toàn độ đa dạng / tránh overfit.")
                    if live is not None:
                        live.next_step()
                    break

            if live is not None:
                live.next_step()

        ckpt.restore_best(gflownet, self.device)

        logger.info("Huấn luyện hoàn tất. Đang chuyển sang chế độ Suy diễn và Đánh giá cuối cùng...")

        holdout_reward = reward_module_holdout if reward_module_holdout is not None else env.reward_module
        if reward_module_holdout is None:
            logger.warning(
                "Không có reward_module_holdout (test set) riêng — chọn tập luật tốt "
                "nhất trên CHÍNH tập val đã dùng để train (kết quả có thể lạc quan hơn thực tế)."
            )

        final_selected, final_report = self._select_best_on_holdout(
            gflownet, env, valid_rules, num_candidates, holdout_reward
        )
        debug_breakdown(final_selected, valid_rules, env.reward_module, logger, label="final_selected (returned)")
        logger.info("Final: %d rules, holdout score=%.4f, reward=%.4f",
                    len(final_selected), final_report["score"], final_report["reward"])

        if live is not None:
            live.summary["best_val_loss"] = ckpt.best_val_loss
            live.summary["n_rules_selected"] = len(final_selected)
            live.summary.update({f"holdout/{k}": v for k, v in final_report.items()})
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
        use_dvc: bool = True,
        dvc_dir: Optional[str] = None,
        sample_weight: Optional[torch.Tensor] = None,
        cover_test: Optional[torch.Tensor] = None,
        correct_test: Optional[torch.Tensor] = None,
        replay_capacity: int = 10000,
        num_candidates: int = 100,
    ) -> List[Rule]:
        """`valid_rules`/`cover`/`correct`/`rule_len` phải đến từ MỘT lần gọi
        duy nhất `RuleValidator.validate_and_build_tensors()` ở ngoài (stage4)
        — pipeline này KHÔNG tự tính lại cover/correct, tránh quét val set
        lần thứ hai (xem README.md, mục "Reward").

        `sample_weight`: (n_val,) float, optional — trọng số uncertainty
        theo mẫu (xem `uncertainty.py`, `compute_sample_weight*`), dùng để
        weighted `f_cover` trong `RuleSetReward` (luật phủ đúng mẫu CNN
        đang yếu sẽ đóng góp coverage cao hơn). PHẢI cùng thứ tự hàng/
        permutation MẪU với `cover`/`correct` gốc TRƯỚC KHI hàm này permute
        lại theo LUẬT bên dưới — permutation đó chỉ hoán vị chiều luật
        (chiều 0), không đụng tới chiều mẫu (chiều 1), nên `sample_weight`
        không cần permute lại theo `perm`.

        `cover_test`/`correct_test`: (n_valid, n_test) bool, optional — cùng
        chiều LUẬT (chiều 0) với `cover`/`correct` nhưng trên tập TEST hoàn
        toàn tách biệt khỏi val. Nếu cung cấp, GIAI ĐOẠN 4 (post-training)
        sẽ chọn tập luật tốt nhất trong `num_candidates` ứng viên sinh ra
        theo reward tính trên tập TEST này (đánh giá khách quan, không lạc
        quan hoá do đã dùng chính val để train). Nếu để None, sẽ dùng tạm
        lại reward trên val kèm cảnh báo log. Không cần truyền `rule_len`
        riêng cho test vì độ dài luật không đổi theo tập dữ liệu.

        `replay_capacity`: dung lượng replay buffer (`gfn.containers.
        ReplayBuffer`, tính theo SỐ QUỸ ĐẠO thật) dùng để huấn luyện
        off-policy ổn định hơn.

        `num_candidates`: số tập luật ứng viên GFlowNet sinh ra ở GIAI ĐOẠN 4
        trước khi chọn tập tốt nhất theo holdout reward.

        `use_dvc`: bật/tắt tracking bằng DVCLive (loss, avg_log_reward,
        grad_norm, lr, logZ mỗi bước train; val loss/mode_diversity mỗi lần
        validate; best_val_loss + n_rules_selected + báo cáo holdout làm
        summary cuối cùng). Cần cài `dvclive` (`pip install dvclive`); nếu
        thiếu, tự động bỏ qua và chỉ log qua logger như trước. `dvc_dir`
        mặc định là `<output_dir>/dvclive`.
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

        if sample_weight is not None:
            sample_weight = sample_weight.to(self.device)
        reward_fn = self._create_reward_function(valid_rules, cover, correct, rule_len, max_rules, sample_weight)
        env = RuleSelectionEnv(n_valid, max_rules, reward_fn, device=self.device)

        # Reward ĐỘC LẬP trên tập test (nếu có) — chỉ hoán vị theo `perm` ở
        # chiều LUẬT (chiều 0) để khớp cover/correct/rule_len, KHÔNG đụng tới
        # chiều mẫu test (chiều 1). Không truyền sample_weight (uncertainty
        # đó chỉ có nghĩa trên mẫu val dùng để train).
        reward_module_holdout = None
        if cover_test is not None and correct_test is not None:
            cover_test = cover_test[perm].to(self.device)
            correct_test = correct_test[perm].to(self.device)
            reward_module_holdout = self._create_reward_function(
                valid_rules, cover_test, correct_test, rule_len, max_rules
            )
            self._last_reward_module = reward_fn  # giữ nguyên tham chiếu reward TRAIN, không bị ghi đè bởi reward TEST

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
                    "alpha": getattr(self, "alpha", 1.0),
                    "lambda_1": getattr(self, "lambda_1", 1.0),
                    "lambda_2": getattr(self, "lambda_2", 1.0),
                    "lambda_3": getattr(self, "lambda_3", 1.0),
                    "K": getattr(self, "K", 10),
                    "gamma": getattr(self, "gamma", 3.0),
                },
                f,
            )
        logger.info("Đã lưu rule order + tensor (khớp index với gflownet_best.pth) tại %s", rule_order_path)

        pf_module = MLP(input_dim=env.state_shape[-1], output_dim=env.n_actions, hidden_dim=gfnet_hidden_dim, n_hidden_layers=2, add_layer_norm = True)
        pb_module = MLP(input_dim=env.state_shape[-1], output_dim=env.n_actions - 1, hidden_dim=gfnet_hidden_dim, n_hidden_layers=2, add_layer_norm = True)

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
            gflownet = FMGFlowNet(logF=pf_estimator)
            optimizer = torch.optim.Adam(gflownet.parameters(), lr=lr, weight_decay=1e-5)
        else:
            raise ValueError(f"loss_type phải là 'tb'/'db'/'fm', nhận '{loss_type}'")

        gflownet.to(self.device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=10, threshold=1e-4, min_lr=1e-6)

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
                        "replay_capacity": replay_capacity,
                        "num_candidates": num_candidates,
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
                num_iterations=num_iterations,
                batch_size=batch_size,
                validation_interval=validation_interval,
                logZ_warmup_steps=logZ_warmup_steps,
                val_samples=val_samples,
                loss_type=loss_type,
                output_dir=output_dir,
                live=live,
                replay_capacity=replay_capacity,
                num_candidates=num_candidates,
                reward_module_holdout=reward_module_holdout,
            )
        except Exception:
            if live is not None:
                live.end()
            raise


class RuleExtractionPipeline(BaseGFlowNetPipeline):
    """U(S) = f_quality + lambda_1*f_cover - lambda_2*f_overlap - lambda_3*f_size,
    R(S) = exp(gamma*U(S)), tính hoàn toàn bằng tensor cover/correct/rule_len
    đã được build sẵn từ bên ngoài. Không sklearn, không proxy net trong
    training loop.

    Khớp với `RuleSetReward` mới trong reward.py:
      - `f_quality` giờ là tổng điểm chất lượng NỘI TẠI mỗi luật
        (q_r = freq_r*(1-err_r)*exp(-alpha*len_r)), không còn accuracy
        chung của cả tập được chọn.
      - `f_overlap` đếm MỌI mẫu bị phủ bởi >1 luật trong S, không phân biệt
        target — nên không còn cần `targets` khi khởi tạo RuleSetReward.
      - `f_size` phạt khi |S| vượt quá `K` luật kỳ vọng.
      - `sample_weight` (tuỳ chọn, từ `uncertainty.py`) làm weighted
        `f_cover` — luật phủ đúng mẫu CNN đang yếu đóng góp coverage cao
        hơn luật chỉ phủ mẫu CNN vốn đã tự tin/đúng.
    """

    def __init__(
        self,
        device: str = "cuda",
        alpha: float = 1.0,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        lambda_3: float = 1.0,
        K: int = 10,
        gamma: float = 3.0,
        grad_clip_max_norm: Optional[float] = 5.0,
    ):
        super().__init__(device, grad_clip_max_norm)
        self.alpha, self.lambda_1, self.lambda_2 = alpha, lambda_1, lambda_2
        self.lambda_3, self.K, self.gamma = lambda_3, K, gamma

    def _reward_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "lambda_3": self.lambda_3,
            "K": self.K,
            "gamma": self.gamma,
        }

    def _create_reward_function(
        self,
        valid_rules: List[Rule],
        cover: torch.Tensor,
        correct: torch.Tensor,
        rule_len: torch.Tensor,
        max_rules: int,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Callable:
        reward_module = RuleSetReward(
            cover=cover,
            correct=correct,
            rule_len=rule_len,
            max_rules=max_rules,
            alpha=self.alpha,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            lambda_3=self.lambda_3,
            K=self.K,
            gamma=self.gamma,
            sample_weight=sample_weight,
        )
        self._last_reward_module = reward_module

        return reward_module
