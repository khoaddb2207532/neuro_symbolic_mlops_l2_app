import pytest
import torch

from src.gflownet.mogfn_pc import (
    ConditionalTBBatch,
    MOGFNPC,
    PreferenceEncoder,
    conditional_tb_loss,
    sample_preferences,
    scalarize_objectives,
)


def test_dirichlet_preferences_live_on_simplex():
    omega = sample_preferences(64, 3, alpha=0.5)
    assert omega.shape == (64, 3)
    assert torch.all(omega > 0)
    assert torch.allclose(omega.sum(-1), torch.ones(64), atol=1e-6)


def test_thermometer_encoding_shape_and_monotonicity():
    encoded = PreferenceEncoder(2, "thermometer", bins=5)(torch.tensor([[0.25, 0.75]]))
    assert encoded.shape == (1, 10)
    assert encoded[0, :5].sum() < encoded[0, 5:].sum()


def test_scalarizations_match_definitions():
    r = torch.tensor([[0.8, 0.5]])
    w = torch.tensor([[0.25, 0.75]])
    assert scalarize_objectives(r, w, "ws").item() == pytest.approx(0.575)
    assert scalarize_objectives(r, w, "wl").item() == pytest.approx(0.8 ** 0.25 * 0.5 ** 0.75)
    assert scalarize_objectives(r, w, "wt").item() == pytest.approx(1 / 0.375)


def test_conditional_model_and_tb_loss_are_differentiable():
    model = MOGFNPC(state_dim=7, n_actions=8, hidden_dim=16, thermometer_bins=4)
    omega = sample_preferences(5, 3)
    assert model.forward_logits(torch.zeros(5, 7), omega).shape == (5, 8)
    assert model.backward_logits(torch.zeros(5, 7), omega).shape == (5, 7)
    loss = conditional_tb_loss(model, ConditionalTBBatch(
        log_pf=torch.randn(5), log_pb=torch.randn(5), log_reward=torch.randn(5), omega=omega
    ))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
