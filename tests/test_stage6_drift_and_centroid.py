import numpy as np
import torch

from src.training.centroid import centroid_push_pull_loss, class_centroids
from src.training.drift import drift_decision, select_relative_drift_threshold


def test_drift_threshold_is_data_driven_and_bounded():
    threshold = select_relative_drift_threshold([0.91, 0.905, 0.90, 0.899], 0.92, 0.10, 0.15)
    assert 0.10 <= threshold <= 0.15
    assert drift_decision(0.80, 0.92, threshold)["refit_required"]


def test_centroids_and_push_pull_loss_are_finite():
    features = torch.tensor([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    centers = class_centroids(features, labels, 2)
    loss = centroid_push_pull_loss(features, labels, centers, margin=1.0)
    assert centers.shape == (2, 2)
    assert np.isfinite(loss.item())
    assert loss.item() >= 0
