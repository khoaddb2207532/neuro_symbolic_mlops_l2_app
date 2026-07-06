"""Môi trường DAG rời rạc để chọn tập luật con bằng GFlowNet (pattern FacesEnv)."""
from typing import Callable, Optional

import torch
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.preprocessors import IdentityPreprocessor


class RuleSelectionEnv(DiscreteEnv):
    """Trạng thái: vector nhị phân {0,1}^n_rules. Hành động {0..n_rules-1}=thêm
    luật i; {n_rules}=exit. Forward mask cấm chọn lại luật đã có hoặc vượt max_rules;
    backward mask cho phép bỏ luật đang được chọn."""

    def __init__(
        self,
        n_rules: int,
        max_rules: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        device: Optional[torch.device] = None,
    ):
        self.n_rules = n_rules
        self.max_rules = max_rules
        self.reward_fn = reward_fn

        device = device if device is not None else torch.device("cpu")
        s0 = torch.zeros(n_rules, dtype=torch.float, device=device)
        sf = torch.full((n_rules,), -1.0, dtype=torch.float, device=device)

        super().__init__(n_actions=n_rules + 1, s0=s0, state_shape=(n_rules,), sf=sf)
        self.preprocessor = IdentityPreprocessor(output_dim=n_rules, target_dtype=torch.float32)

    def make_states_class(self) -> type:
        env = self

        class RuleStates(super().make_states_class()):
            def _compute_forward_masks(self) -> torch.Tensor:
                state = self.tensor
                selected = state.sum(dim=-1, keepdim=True)
                at_max = selected >= env.max_rules

                masks = torch.zeros(
                    self.batch_shape + (env.n_actions,), dtype=torch.bool, device=self.device
                )
                can_add = (state == 0) & ~at_max
                masks[..., : env.n_rules] = can_add
                masks[..., env.n_rules] = True
                return masks

            def _compute_backward_masks(self) -> torch.Tensor:
                return self.tensor != 0

        return RuleStates

    @staticmethod
    def _action_index(states_tensor: torch.Tensor, actions_tensor: torch.Tensor) -> torch.Tensor:
        act = actions_tensor
        while act.dim() < states_tensor.dim():
            act = act.unsqueeze(-1)
        return act

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        idx = self._action_index(states.tensor, actions.tensor)
        new_tensor = states.tensor.scatter(-1, idx, 1, reduce="add").clamp(0, 1)
        return self.States(new_tensor)

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        idx = self._action_index(states.tensor, actions.tensor)
        new_tensor = states.tensor.scatter(-1, idx, -1, reduce="add").clamp(0, 1)
        return self.States(new_tensor)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        return self.reward_fn(final_states.tensor)

    def log_reward(self, final_states: DiscreteStates) -> torch.Tensor:
        raw = self.reward_fn.reward_module.score(final_states.tensor)  # đã ở thang cố định (trước khi exp)
        return self.reward_fn.reward_module.beta * raw
