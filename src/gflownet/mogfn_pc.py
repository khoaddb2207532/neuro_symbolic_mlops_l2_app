"""Preference-Conditional Multi-Objective GFlowNet building blocks.

The module is deliberately independent from torchgfn's scalar ``logZ`` because
MOGFN-PC requires both policies and the partition function to depend on omega.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn


Scalarization = Literal["ws", "wt", "wl"]
PreferenceEncoding = Literal["vanilla", "thermometer"]


def sample_preferences(
    batch_size: int,
    n_objectives: int,
    alpha: float | torch.Tensor = 1.0,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Draw omega ~ Dirichlet(alpha) on the d-dimensional simplex."""
    concentration = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    if concentration.ndim == 0:
        concentration = concentration.repeat(n_objectives)
    if concentration.shape != (n_objectives,):
        raise ValueError("alpha must be a scalar or have shape (n_objectives,)")
    if torch.any(concentration <= 0):
        raise ValueError("Dirichlet concentration parameters must be positive")
    return torch.distributions.Dirichlet(concentration).sample((batch_size,))


class PreferenceEncoder(nn.Module):
    """Vanilla or differentiable thermometer encoding of preference vectors."""

    def __init__(self, n_objectives: int, mode: PreferenceEncoding = "vanilla", bins: int = 16):
        super().__init__()
        if mode not in ("vanilla", "thermometer"):
            raise ValueError("mode must be 'vanilla' or 'thermometer'")
        if bins < 2:
            raise ValueError("thermometer bins must be >= 2")
        self.n_objectives, self.mode, self.bins = n_objectives, mode, bins
        self.register_buffer("thresholds", torch.linspace(0.0, 1.0, bins))

    @property
    def output_dim(self) -> int:
        return self.n_objectives if self.mode == "vanilla" else self.n_objectives * self.bins

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        if omega.shape[-1] != self.n_objectives:
            raise ValueError(f"omega must end in {self.n_objectives} objectives")
        if self.mode == "vanilla":
            return omega
        return (omega.unsqueeze(-1) >= self.thresholds).to(omega.dtype).flatten(start_dim=-2)


def scalarize_objectives(
    objectives: torch.Tensor,
    omega: torch.Tensor,
    method: Scalarization = "wl",
    *,
    ideal: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Combine positive, maximized objectives into one reward.

    WT is returned as ``1 / (eps + max_i omega_i |R_i-z_i*|)`` so that, like
    WS/WL and GFlowNet rewards, larger values are always better.
    """
    if objectives.shape[-1] != omega.shape[-1]:
        raise ValueError("objectives and omega must have the same final dimension")
    omega = omega.to(device=objectives.device, dtype=objectives.dtype)
    if method == "ws":
        return (omega * objectives).sum(-1).clamp_min(eps)
    if method == "wl":
        return torch.exp((omega * objectives.clamp_min(eps).log()).sum(-1)).clamp_min(eps)
    if method == "wt":
        if ideal is None:
            ideal = torch.ones(objectives.shape[-1], device=objectives.device, dtype=objectives.dtype)
        distance = (omega * (objectives - ideal).abs()).amax(-1)
        return 1.0 / (distance + eps)
    raise ValueError("method must be one of: ws, wt, wl")


class ConditionalMLP(nn.Module):
    """MLP shared across every preference-conditioned subproblem."""

    def __init__(self, state_dim: int, output_dim: int, encoder: PreferenceEncoder,
                 hidden_dim: int = 256, n_hidden_layers: int = 2):
        super().__init__()
        self.encoder = encoder
        layers: list[nn.Module] = []
        width = state_dim + encoder.output_dim
        for _ in range(n_hidden_layers):
            layers.extend((nn.Linear(width, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)))
            width = hidden_dim
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state.float(), self.encoder(omega)), dim=-1))


class MOGFNPC(nn.Module):
    """Single conditional forward/backward policy plus conditional log Z."""

    def __init__(self, state_dim: int, n_actions: int, n_objectives: int = 3,
                 hidden_dim: int = 256, preference_encoding: PreferenceEncoding = "thermometer",
                 thermometer_bins: int = 16):
        super().__init__()
        pf_encoder = PreferenceEncoder(n_objectives, preference_encoding, thermometer_bins)
        pb_encoder = PreferenceEncoder(n_objectives, preference_encoding, thermometer_bins)
        z_encoder = PreferenceEncoder(n_objectives, preference_encoding, thermometer_bins)
        self.pf = ConditionalMLP(state_dim, n_actions, pf_encoder, hidden_dim)
        self.pb = ConditionalMLP(state_dim, n_actions - 1, pb_encoder, hidden_dim)
        self.log_z = ConditionalMLP(0, 1, z_encoder, hidden_dim, n_hidden_layers=1)

    def forward_logits(self, state: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        return self.pf(state, omega)

    def backward_logits(self, state: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        return self.pb(state, omega)

    def log_partition(self, omega: torch.Tensor) -> torch.Tensor:
        empty = omega.new_empty((*omega.shape[:-1], 0))
        return self.log_z(empty, omega).squeeze(-1)


@dataclass
class ConditionalTBBatch:
    log_pf: torch.Tensor
    log_pb: torch.Tensor
    log_reward: torch.Tensor
    omega: torch.Tensor


def conditional_tb_loss(model: MOGFNPC, batch: ConditionalTBBatch) -> torch.Tensor:
    """Squared conditional Trajectory Balance residual, averaged over a batch."""
    residual = model.log_partition(batch.omega) + batch.log_pf - batch.log_reward - batch.log_pb
    return residual.square().mean()


def sample_rule_trajectories(model: MOGFNPC, reward_module, batch_size: int,
                             max_rules: int, alpha: float = 1.0,
                             scalarization: Scalarization = "wl",
                             exploration_delta: float = 0.0,
                             omega: Optional[torch.Tensor] = None):
    """Sample subset-building trajectories and retain all TB log-probabilities."""
    device = next(model.parameters()).device
    n_rules = model.pf.net[-1].out_features - 1
    omega = sample_preferences(batch_size, 3, alpha, device=device) if omega is None else omega.to(device)
    state = torch.zeros(batch_size, n_rules, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    log_pf = torch.zeros(batch_size, device=device)
    log_pb = torch.zeros(batch_size, device=device)
    for _ in range(max_rules + 1):
        active = ~done
        if not active.any():
            break
        ids = active.nonzero(as_tuple=False).squeeze(-1)
        current, pref = state[ids], omega[ids]
        logits = model.forward_logits(current, pref)
        can_add = (current == 0) & (current.sum(-1, keepdim=True) < max_rules)
        mask = torch.cat((can_add, torch.ones(len(ids), 1, dtype=torch.bool, device=device)), -1)
        policy = torch.softmax(logits.masked_fill(~mask, -torch.inf), -1)
        uniform = mask.float() / mask.float().sum(-1, keepdim=True)
        probs = (1 - exploration_delta) * policy + exploration_delta * uniform
        action = torch.multinomial(probs, 1).squeeze(-1)
        # Exploration changes only the behaviour distribution; TB is evaluated
        # under the learned forward policy P_F.
        log_pf[ids] += policy.gather(1, action[:, None]).squeeze(1).clamp_min(1e-12).log()
        exits = action == n_rules
        done[ids[exits]] = True
        add_ids, add_actions = ids[~exits], action[~exits]
        if len(add_ids):
            next_state = state[add_ids].clone()
            next_state.scatter_(1, add_actions[:, None], 1.0)
            pb_logits = model.backward_logits(next_state, omega[add_ids])
            pb_probs = torch.softmax(pb_logits.masked_fill(~next_state.bool(), -torch.inf), -1)
            log_pb[add_ids] += pb_probs.gather(1, add_actions[:, None]).squeeze(1).clamp_min(1e-12).log()
            state[add_ids] = next_state
    log_reward = reward_module.conditional_log_reward(state, omega, scalarization)
    return state, omega, ConditionalTBBatch(log_pf, log_pb, log_reward, omega)


def pareto_mask(objectives: torch.Tensor) -> torch.Tensor:
    """Return the non-dominated mask for maximization objectives."""
    dominates = (objectives[:, None] >= objectives[None]).all(-1)
    strictly = (objectives[:, None] > objectives[None]).any(-1)
    return ~(dominates & strictly).any(dim=0)
