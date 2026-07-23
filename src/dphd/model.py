"""The hyperdimensional classifier (Sections 3.1.2 - 3.1.4).

Training bundles (sums) the hypervectors of each class into a single class
hypervector; retraining walks the training set and moves misclassified
hypervectors from the wrong class to the right one; inference is a dot product
against every class hypervector.
"""

from __future__ import annotations

import torch


def train_class_hvs(
    encoded: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Bundle hypervectors per class: C_s = sum of H over class s. Eq. (1)."""
    class_hvs = torch.zeros(
        (n_classes, encoded.shape[1]), dtype=encoded.dtype, device=encoded.device
    )
    class_hvs.index_add_(0, labels, encoded)
    return class_hvs


def retrain(
    class_hvs: torch.Tensor,
    encoded: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Perceptron-style error-driven refinement. Eq. (3).

    On a misprediction the sample is added to its true class hypervector and
    subtracted from the predicted one. Updates are applied one sample at a time
    (order shuffled each epoch), which is what makes this an online rule rather
    than a batch one.
    """
    class_hvs = class_hvs.clone()
    n = encoded.shape[0]
    for _ in range(n_epochs):
        order = torch.randperm(n, device=encoded.device, generator=generator)
        for idx in order:
            h = encoded[idx]
            guess = torch.matmul(class_hvs, h).argmax()
            true = labels[idx]
            if guess != true:
                class_hvs[true] += h
                class_hvs[guess] -= h
    return class_hvs


def predict(encoded: torch.Tensor, class_hvs: torch.Tensor) -> torch.Tensor:
    """Assign each query to the most similar class hypervector. Eq. (2).

    The query norm is constant across classes and the class norms are fixed at
    inference time, so the cosine reduces to a dot product.
    """
    return (encoded @ class_hvs.T).argmax(dim=1)


def accuracy(
    encoded: torch.Tensor, class_hvs: torch.Tensor, labels: torch.Tensor
) -> float:
    """Top-1 accuracy in [0, 1]."""
    return (predict(encoded, class_hvs) == labels).float().mean().item()


def mean_class_similarity(encoded: torch.Tensor, labels: torch.Tensor) -> float:
    """Average cosine similarity between same-class training hypervectors.

    This is ``mu_c`` in Eq. (12)-(13): the per-sample contribution to the signal
    term of the SNR. Computed exactly, without materialising the pairwise
    matrix, via ||sum of unit vectors||^2 = n + sum_{i != j} cos(h_i, h_j).
    """
    normed = encoded / encoded.norm(dim=1, keepdim=True).clamp_min(1e-12)
    total, pairs = 0.0, 0
    for s in torch.unique(labels):
        block = normed[labels == s]
        n = block.shape[0]
        if n < 2:
            continue
        off_diag_sum = block.sum(dim=0).pow(2).sum().item() - n
        total += off_diag_sum
        pairs += n * (n - 1)
    return total / pairs if pairs else 0.0
