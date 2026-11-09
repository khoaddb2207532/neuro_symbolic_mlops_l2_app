"""Bayesian rule marginalization trên archive state của GFlowNet EliteTracker."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.rules.bayesian_penalty import BayesianRuleMarginalization
from src.rules.rule_types import Rule


ELITE_STATE_FILENAMES: Tuple[str, ...] = (
    "gflownet_elite_states.pt",
    "elite_states.pt",
    "elite_states.pth",
    "elite_states.pkl",
)


def rule_order_fingerprint(valid_rules: Sequence[Rule]) -> str:
    payload = "\n".join(repr(rule) for rule in valid_rules).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_elite_states(
    explicit_path: Optional[str], search_roots: Sequence[str | Path]
) -> Path:
    """Tìm artifact state, ưu tiên path cấu hình rồi mới quét các input root."""
    if explicit_path:
        explicit = Path(explicit_path).expanduser()
        if explicit.is_file():
            return explicit.resolve()
        if explicit.is_dir():
            search_roots = (explicit, *search_roots)
        else:
            raise FileNotFoundError(f"Không tìm thấy elite states_path: {explicit}")

    checked: List[str] = []
    for root_value in search_roots:
        root = Path(root_value).expanduser()
        checked.append(str(root))
        if not root.exists():
            continue
        if root.is_file() and root.name in ELITE_STATE_FILENAMES:
            return root.resolve()
        candidates: List[Path] = []
        for name in ELITE_STATE_FILENAMES:
            candidates.extend(root.rglob(name))
        unique = sorted(
            {path.resolve() for path in candidates}, key=lambda path: str(path)
        )
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            rendered = "\n- ".join(map(str, unique))
            raise RuntimeError(
                f"Tìm thấy nhiều elite-state artifact trong {root}; hãy đặt "
                "rule_penalty_bayesian_elite.states_path để chọn đúng file:\n- "
                + rendered
            )

    raise FileNotFoundError(
        "Không tìm thấy elite-state artifact. Đã kiểm tra: " + ", ".join(checked)
    )


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as file:
            return pickle.load(file)
    return torch.load(path, map_location="cpu")


def load_elite_states(
    path: str | Path, valid_rules: Sequence[Rule]
) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
    """Nạp và xác thực state trước khi bất kỳ batch CNN nào được train."""
    source = Path(path)
    payload = _load_payload(source)
    metadata: dict = payload if isinstance(payload, dict) else {}

    if isinstance(payload, dict):
        states = next(
            (
                payload[key]
                for key in ("states", "elite_states", "terminating_states", "masks")
                if key in payload
            ),
            None,
        )
        rewards = next(
            (
                payload[key]
                for key in ("log_rewards", "rewards", "scores")
                if key in payload
            ),
            None,
        )
    else:
        states, rewards = payload, None

    if states is None:
        raise ValueError(f"{source} không chứa states/elite_states/terminating_states/masks.")
    states = torch.as_tensor(states)
    if states.ndim == 1:
        states = states.unsqueeze(0)
    if states.ndim != 2:
        raise ValueError(f"Elite states phải là tensor 2-D, nhận shape={tuple(states.shape)}.")
    if states.shape[0] == 0:
        raise ValueError("Elite-state artifact rỗng.")
    if states.shape[1] != len(valid_rules):
        raise ValueError(
            f"Elite states có {states.shape[1]} cột nhưng rule order có "
            f"{len(valid_rules)} luật. Không thể ánh xạ state -> rule an toàn."
        )
    if not torch.all((states == 0) | (states == 1)):
        raise ValueError("Elite states chỉ được chứa giá trị nhị phân 0/1.")
    states = states.bool().cpu()

    expected_fingerprint = rule_order_fingerprint(valid_rules)
    stored_fingerprint = metadata.get("rule_order_fingerprint")
    if stored_fingerprint and stored_fingerprint != expected_fingerprint:
        raise ValueError(
            "Elite states không khớp thứ tự luật trong gflownet_rule_order.pkl "
            "(rule_order_fingerprint khác nhau)."
        )
    stored_n_rules = metadata.get("n_rules")
    if stored_n_rules is not None and int(stored_n_rules) != len(valid_rules):
        raise ValueError("Metadata n_rules của elite states không khớp rule order.")

    if rewards is not None:
        rewards = torch.as_tensor(rewards, dtype=torch.float32).flatten().cpu()
        if rewards.numel() != states.shape[0]:
            raise ValueError("Số log_rewards/scores không khớp số elite states.")
        if not torch.isfinite(rewards).all():
            raise ValueError("log_rewards/scores của elite states chứa NaN/Inf.")

    # Loại state trùng. Giữ reward lớn nhất của mỗi state nếu artifact có score.
    unique, inverse = torch.unique(states, dim=0, return_inverse=True)
    if rewards is not None:
        dedup_rewards = torch.full((unique.shape[0],), float("-inf"))
        dedup_rewards.scatter_reduce_(
            0, inverse, rewards, reduce="amax", include_self=True
        )
        rewards = dedup_rewards
    return unique, rewards, {
        "source_path": str(source.resolve()),
        "n_states": int(unique.shape[0]),
        "n_rules": int(unique.shape[1]),
        "has_log_rewards": rewards is not None,
        "rule_order_fingerprint": expected_fingerprint,
    }


class _EliteStateSampler(nn.Module):
    def __init__(
        self,
        states: torch.Tensor,
        log_rewards: Optional[torch.Tensor] = None,
        reward_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if reward_temperature <= 0:
            raise ValueError("reward_temperature phải > 0.")
        self.register_buffer("states", states.bool())
        if log_rewards is None:
            probabilities = torch.full(
                (states.shape[0],), 1.0 / states.shape[0], dtype=torch.float32
            )
        else:
            probabilities = torch.softmax(log_rewards / reward_temperature, dim=0)
        self.register_buffer("probabilities", probabilities)

    @torch.no_grad()
    def sample_trajectories(self, env, n: int, save_logprobs: bool = False):
        del env, save_logprobs
        indices = torch.multinomial(self.probabilities, n, replacement=True)
        masks = self.states.index_select(0, indices)
        return SimpleNamespace(
            terminating_states=SimpleNamespace(tensor=masks)
        )


class EliteStateBayesianMarginalization(BayesianRuleMarginalization):
    """Drop-in penalty lấy mẫu từ elite-state archive thay cho policy checkpoint."""

    def __init__(
        self,
        *,
        valid_rules: List[Rule],
        states: torch.Tensor,
        log_rewards: Optional[torch.Tensor] = None,
        reward_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        sampler = _EliteStateSampler(states, log_rewards, reward_temperature)
        env = SimpleNamespace(n_rules=len(valid_rules))
        super().__init__(
            valid_rules=valid_rules,
            gflownet=sampler,
            env=env,
            **kwargs,
        )
