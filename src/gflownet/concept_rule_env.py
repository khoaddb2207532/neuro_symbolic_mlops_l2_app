"""Môi trường GFlowNet XÂY DỰNG từng luật trên image_concept (torchgfn DiscreteEnv).

Khác với ``RuleSelectionEnv`` (chọn tập con từ luật có sẵn), env này sinh luật
từng bước theo DAG đơn điệu (trạng thái chỉ được bật thêm bit, không bao giờ tắt
ở forward):

  1. Thêm literal: "concept k có mặt" hoặc "concept k vắng mặt"  (tối đa max_literals)
  2. Chọn vế phải: "=> class c"  (đúng 1 lần, chỉ sau khi đã có ít nhất 1 literal)
  3. Exit: kết thúc luật (chỉ sau khi đã chọn vế phải)

Layout trạng thái (D = 2K + C):
  slot 2k     : concept k có mặt
  slot 2k + 1 : concept k vắng mặt
  slot 2K + c : vế phải "=> class c"

Action i ∈ [0, D) bật slot i; action D là exit. Ràng buộc:
  - không bật cả hai literal của cùng một concept (mâu thuẫn);
  - không thêm literal sau khi đã có vế phải;
  - backward chỉ gỡ literal khi chưa có vế phải (đảo ngược đúng thứ tự xây dựng).
"""
from typing import Optional

import torch
import torch.nn.functional as F
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.preprocessors import IdentityPreprocessor
from gfn.states import DiscreteStates

from src.gflownet.concept_rule_reward import ConceptRuleReward


class ConceptRuleEnv(DiscreteEnv):
    def __init__(
        self,
        n_concepts: int,
        num_classes: int,
        max_literals: int,
        reward_module: ConceptRuleReward,
        device: Optional[torch.device] = None,
    ) -> None:
        if n_concepts < 1 or num_classes < 2 or max_literals < 1:
            raise ValueError("Cần n_concepts>=1, num_classes>=2, max_literals>=1.")
        self.n_concepts = n_concepts
        self.num_classes = num_classes
        self.max_literals = max_literals
        self.reward_module = reward_module
        self.dim = 2 * n_concepts + num_classes

        device = device if device is not None else torch.device("cpu")
        s0 = torch.zeros(self.dim, dtype=torch.float, device=device)
        sf = torch.full((self.dim,), -1.0, dtype=torch.float, device=device)

        super().__init__(n_actions=self.dim + 1, s0=s0, state_shape=(self.dim,), sf=sf)
        self.preprocessor = IdentityPreprocessor(output_dim=self.dim, target_dtype=torch.float32)

    def make_states_class(self) -> type:
        env = self
        K = self.n_concepts

        class ConceptRuleStates(super().make_states_class()):
            def _compute_forward_masks(self) -> torch.Tensor:
                state = self.tensor
                lit = state[..., : 2 * K]
                head = state[..., 2 * K:]
                n_lit = lit.sum(dim=-1, keepdim=True)
                has_head = head.sum(dim=-1, keepdim=True) > 0

                # Concept k đã có literal (bất kỳ polarity) -> cấm cả hai slot của k.
                used = lit.reshape(*lit.shape[:-1], K, 2).sum(dim=-1) > 0          # (..., K)
                used_slot = used.unsqueeze(-1).expand(*used.shape, 2).reshape(*used.shape[:-1], 2 * K)

                can_add_literal = (~has_head) & (n_lit < env.max_literals)         # (..., 1)
                literal_ok = (lit == 0) & (~used_slot) & can_add_literal          # (..., 2K)
                head_ok = ((~has_head) & (n_lit > 0)).expand(*head.shape)         # (..., C)
                exit_ok = has_head                                                 # (..., 1)
                return torch.cat([literal_ok, head_ok, exit_ok], dim=-1)

            def _compute_backward_masks(self) -> torch.Tensor:
                state = self.tensor
                lit = state[..., : 2 * K]
                head = state[..., 2 * K:]
                has_head = head.sum(dim=-1, keepdim=True) > 0
                literal_back = (lit != 0) & (~has_head)
                head_back = head != 0
                return torch.cat([literal_back, head_back], dim=-1)

        return ConceptRuleStates

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        idx = actions.tensor.long().reshape(states.tensor.shape[:-1])
        onehot = F.one_hot(idx, self.dim).to(states.tensor.dtype)
        return self.States(torch.clamp(states.tensor + onehot, max=1.0))

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        idx = actions.tensor.long().reshape(states.tensor.shape[:-1])
        onehot = F.one_hot(idx, self.dim).to(states.tensor.dtype)
        return self.States(torch.clamp(states.tensor - onehot, min=0.0))

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        return torch.exp(self.log_reward(final_states))

    def log_reward(self, final_states: DiscreteStates) -> torch.Tensor:
        return self.reward_module.log_reward(final_states.tensor)
