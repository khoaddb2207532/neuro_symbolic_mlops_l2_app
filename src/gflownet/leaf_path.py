"""RF leaf-path GFlowNet implemented with torchgfn 2.4.x.

The project defines only its domain environment (valid RF transitions and rewards).
Sampling, estimators, log-Z and Trajectory Balance are provided by torchgfn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

import torch
from torch import nn

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import TBGFlowNet
from gfn.preprocessors import OneHotPreprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States
from gfn.utils.modules import MLP

from src.rules.rule_types import Rule


def _condition_key(condition) -> Tuple[int, str, float]:
    return int(condition.feature_index), condition.operator, round(float(condition.threshold), 10)


@dataclass
class PathNode:
    parent: int | None
    parent_action: int | None
    token: tuple | None
    children: Dict[tuple, int] = field(default_factory=dict)
    leaf_id: int | None = None


class LeafPathGraph:
    """One artificial root followed by real RF tree/split edges."""

    def __init__(self, rules_by_id: Mapping[int, Rule]):
        self.nodes: List[PathNode] = [PathNode(None, None, None)]
        self.leaf_to_node: Dict[int, int] = {}
        for leaf_id, rule in sorted(rules_by_id.items()):
            tree_id = getattr(rule, "tree_id", -1)
            if tree_id < 0:
                raise ValueError("RF rule has no tree_id; rerun stage 3 with the updated RuleExtractor")
            current = 0
            tokens = [("tree", int(tree_id))] + [_condition_key(c) for c in rule.conditions]
            for token in tokens:
                child = self.nodes[current].children.get(token)
                if child is None:
                    action = len(self.nodes[current].children)
                    child = len(self.nodes)
                    self.nodes[current].children[token] = child
                    self.nodes.append(PathNode(current, action, token))
                current = child
            if self.nodes[current].leaf_id is not None:
                raise ValueError("two filtered leaves have the same RF path")
            self.nodes[current].leaf_id = int(leaf_id)
            self.leaf_to_node[int(leaf_id)] = current
        if not self.leaf_to_node:
            raise ValueError("leaf-path graph requires at least one terminal")
        self.max_branching = max(len(node.children) for node in self.nodes)

    def path_edges(self, leaf_id: int) -> List[Tuple[int, int, int]]:
        child = self.leaf_to_node[leaf_id]
        reverse = []
        while self.nodes[child].parent is not None:
            node = self.nodes[child]
            reverse.append((int(node.parent), int(node.parent_action), child))
            child = int(node.parent)
        return list(reversed(reverse))


class TorchGFNLeafPathEnv(DiscreteEnv):
    """torchgfn DiscreteEnv whose only terminal states are filtered RF leaves."""

    def __init__(self, graph: LeafPathGraph, rewards: Mapping[int, float], device: str):
        self.graph = graph
        self.node_rewards = {graph.leaf_to_node[i]: max(float(r), 1e-12) for i, r in rewards.items()}
        self.transition_table = {}
        self.inverse_table = {}
        for parent, node in enumerate(graph.nodes):
            for action, child in enumerate(node.children.values()):
                self.transition_table[(parent, action)] = child
                self.inverse_table[(child, action)] = parent
        s0 = torch.tensor([0], dtype=torch.long, device=device)
        sf = torch.tensor([-1], dtype=torch.long, device=device)
        self.s0, self.sf = s0, sf
        super().__init__(graph.max_branching + 1, s0, (1,), sf=sf, debug=False)

    def make_states_class(self) -> type[DiscreteStates]:
        env = self

        class LeafPathStates(DiscreteStates):
            state_shape = (1,)
            s0, sf = env.s0, env.sf
            make_random_states = env.make_random_states
            n_actions = env.n_actions

            def _compute_forward_masks(self):
                masks = torch.zeros((*self.batch_shape, env.n_actions), dtype=torch.bool, device=self.device)
                flat = self.tensor.reshape(-1).tolist()
                view = masks.reshape(-1, env.n_actions)
                for row, node_id in enumerate(flat):
                    if node_id >= 0 and env.graph.nodes[node_id].leaf_id is not None:
                        view[row, env.n_actions - 1] = True
                    elif node_id >= 0:
                        view[row, : len(env.graph.nodes[node_id].children)] = True
                return masks

            def _compute_backward_masks(self):
                masks = torch.zeros((*self.batch_shape, env.n_actions - 1), dtype=torch.bool, device=self.device)
                flat = self.tensor.reshape(-1).tolist()
                view = masks.reshape(-1, env.n_actions - 1)
                for row, node_id in enumerate(flat):
                    if node_id > 0:
                        view[row, env.graph.nodes[node_id].parent_action] = True
                return masks

        return LeafPathStates

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        pairs = zip(states.tensor.reshape(-1).tolist(), actions.tensor.reshape(-1).tolist())
        values = [self.transition_table[(int(state), int(action))] for state, action in pairs]
        return self.States(torch.tensor(values, device=self.device, dtype=torch.long).reshape(-1, 1))

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        pairs = zip(states.tensor.reshape(-1).tolist(), actions.tensor.reshape(-1).tolist())
        values = [self.inverse_table[(int(state), int(action))] for state, action in pairs]
        return self.States(torch.tensor(values, device=self.device, dtype=torch.long).reshape(-1, 1))

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        return torch.tensor([self.node_rewards[int(i)] for i in final_states.tensor.reshape(-1).tolist()],
                            device=self.device, dtype=torch.get_default_dtype())

    def get_states_indices(self, states: States) -> torch.Tensor:
        return states.tensor.long().squeeze(-1)


class TorchGFNLeafPathModel(nn.Module):
    """Serializable wrapper around torchgfn's TBGFlowNet and domain environment."""

    def __init__(self, gflownet: TBGFlowNet, env: TorchGFNLeafPathEnv):
        super().__init__()
        self.gflownet, self.env = gflownet, env

    def terminal_probabilities(self) -> Dict[int, torch.Tensor]:
        probabilities = {}
        for leaf_id in self.env.graph.leaf_to_node:
            value = next(self.gflownet.parameters()).new_tensor(1.0)
            for parent, action, _ in self.env.graph.path_edges(leaf_id):
                states = self.env.states_from_tensor(torch.tensor([[parent]], device=self.env.device))
                output = self.gflownet.pf(states)
                distribution = self.gflownet.pf.to_probability_distribution(states, output)
                value = value * distribution.probs[0, action]
            probabilities[leaf_id] = value
        normalizer = torch.stack(list(probabilities.values())).sum().clamp_min(1e-12)
        return {leaf_id: value / normalizer for leaf_id, value in probabilities.items()}


def train_leaf_path_gflownet(rules_by_id: Mapping[int, Rule], rewards: Mapping[int, float], *,
                             hidden_dim: int, n_layers: int, learning_rate: float,
                             steps: int, exploration: float, reward_beta: float,
                             kl_patience: int, kl_tolerance: float, seed: int,
                             device: str, batch_size: int = 64, logz_lr: float = 0.05):
    """Train using torchgfn Sampler and TBGFlowNet loss, not a local TB implementation."""
    torch.manual_seed(seed)
    graph = LeafPathGraph(rules_by_id)
    tempered_rewards = {i: max(float(value), 1e-12) ** reward_beta for i, value in rewards.items()}
    env = TorchGFNLeafPathEnv(graph, tempered_rewards, device)
    preprocessor = OneHotPreprocessor(len(graph.nodes), env.get_states_indices)
    module = MLP(preprocessor.output_dim, env.n_actions, hidden_dim=hidden_dim,
                 n_hidden_layers=n_layers, add_layer_norm=True)
    backward_module = MLP(preprocessor.output_dim, env.n_actions - 1, hidden_dim=hidden_dim,
                          n_hidden_layers=n_layers, add_layer_norm=True)
    pf = DiscretePolicyEstimator(module, env.n_actions, preprocessor=preprocessor, is_backward=False)
    pb = DiscretePolicyEstimator(backward_module, env.n_actions, preprocessor=preprocessor,
                                 is_backward=True)
    # P_B has one valid action per state in this trie; keeping the estimator makes
    # the PF/PB architecture explicit and checkpointed as required by the protocol.
    gflownet = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0).to(device)
    optimizer = torch.optim.Adam([
        {"params": list(gflownet.pf_pb_parameters()), "lr": learning_rate},
        {"params": list(gflownet.logz_parameters()), "lr": logz_lr},
    ])
    sampler = Sampler(estimator=pf)
    wrapper = TorchGFNLeafPathModel(gflownet, env)
    leaf_ids = sorted(rules_by_id)
    target = torch.tensor([tempered_rewards[i] for i in leaf_ids], device=device)
    target = target / target.sum()
    best_kl, stale, completed, loss_tail = float("inf"), 0, 0, []
    for step in range(steps):
        trajectories = sampler.sample_trajectories(env, n=batch_size, epsilon=exploration,
                                                   save_logprobs=False, save_estimator_outputs=False)
        loss = gflownet.loss_from_trajectories(env, trajectories, recalculate_all_logprobs=True)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 5.0)
        optimizer.step()
        loss_tail.append(float(loss.detach().cpu()))
        loss_tail = loss_tail[-50:]
        completed = step + 1
        if completed % 25 == 0:
            with torch.no_grad():
                learned_map = wrapper.terminal_probabilities()
                learned = torch.stack([learned_map[i] for i in leaf_ids]).clamp_min(1e-12)
                kl = float(torch.sum(target * (target.log() - learned.log())).item())
            if best_kl - kl > kl_tolerance:
                best_kl, stale = kl, 0
            else:
                stale += 1
            if stale >= kl_patience:
                break
    with torch.no_grad():
        final_map = wrapper.terminal_probabilities()
        final_probs = torch.stack([final_map[i] for i in leaf_ids]).clamp_min(1e-12)
        entropy = float(-(final_probs * final_probs.log()).sum().item())
    loss_mean = sum(loss_tail) / max(len(loss_tail), 1)
    loss_std = (sum((value - loss_mean) ** 2 for value in loss_tail) / max(len(loss_tail), 1)) ** 0.5
    return wrapper, {"steps": completed, "best_kl": best_kl, "n_leaves": len(leaf_ids),
                     "final_tb_loss_mean": loss_mean, "final_tb_loss_std": loss_std,
                     "tb_loss_relative_std": loss_std / max(abs(loss_mean), 1e-12),
                     "leaf_entropy": entropy, "converged": stale >= kl_patience,
                     "torchgfn": True}
