"""Push-pull centroid baseline used only by the final ablation."""
import torch
import torch.nn.functional as F


def centroid_push_pull_loss(features, labels, centroids, margin=1.0):
    """Pull features to their class centroid and push them from the nearest rival."""
    distances = torch.cdist(features, centroids)
    own = distances.gather(1, labels[:, None]).squeeze(1)
    rival = distances.masked_fill(
        F.one_hot(labels, centroids.shape[0]).bool(), float("inf")
    ).min(dim=1).values
    return own.pow(2).mean() + F.relu(margin - rival).pow(2).mean()


def class_centroids(features, labels, num_classes):
    centers = []
    for class_id in range(num_classes):
        selected = features[labels == class_id]
        if not len(selected):
            raise ValueError(f"class {class_id} has no training feature")
        centers.append(selected.mean(dim=0))
    return torch.stack(centers)
