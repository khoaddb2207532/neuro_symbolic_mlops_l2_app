from types import SimpleNamespace

import pytest
import torch

from src.gflownet.pipeline import _EliteTracker


def _trajectories(masks, rewards):
    return SimpleNamespace(
        log_rewards=torch.tensor(rewards, dtype=torch.float32),
        terminating_states=SimpleNamespace(
            tensor=torch.tensor(masks, dtype=torch.bool)
        ),
    )


def test_elite_checkpoint_tracks_ruleset_reward_not_convergence(tmp_path):
    path = tmp_path / "gflownet_best_elite.pth"
    tracker = _EliteTracker(str(path))
    model = torch.nn.Linear(2, 2)
    rules = ["r0", "r1", "r2"]

    improved = tracker.update(
        [_trajectories([[1, 0, 0], [0, 1, 1]], [0.2, 0.9])],
        rules,
    )
    assert improved
    tracker.save_checkpoint(
        model, iteration=17, n_valid=3, max_rules=2
    )

    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["checkpoint_role"] == "elite"
    assert checkpoint["iteration"] == 17
    assert checkpoint["best_log_reward"] == pytest.approx(0.9)
    assert checkpoint["elite_mask"].tolist() == [False, True, True]
    assert tracker.best_selected == ["r1", "r2"]

    assert not tracker.update(
        [_trajectories([[1, 1, 0]], [0.5])], rules
    )
