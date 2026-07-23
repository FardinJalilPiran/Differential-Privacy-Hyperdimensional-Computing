"""Differential privacy for the HD model (Sections 3.2, 3.3, 3.3.1).

Privacy is enforced by adding Gaussian noise to the *class hypervectors* after
training, so what leaves the training environment is already private. The scale
of that noise follows the standard Gaussian mechanism, and the SNR expression
below is what lets you predict the accuracy cost of a given epsilon *before*
paying it.
"""

from __future__ import annotations

import math

import torch


def dp_sigma(epsilon: float, delta: float = 1e-4) -> float:
    """Noise multiplier satisfying (epsilon, delta)-DP. Theorem 1, Eq. (7).

    ``delta`` should be smaller than the inverse of the dataset size; the paper
    uses 1e-4.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    return math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def sensitivity(encoded: torch.Tensor) -> float:
    """Model sensitivity: the largest L2 norm over encoded training samples.

    Eq. (9). Adding or removing one sample changes a class hypervector by at
    most this much, so it sets the scale of the noise that must be added.
    """
    return encoded.norm(p=2, dim=1).max().item()


def privatize(
    class_hvs: torch.Tensor,
    epsilon: float,
    delta_f: float,
    delta: float = 1e-4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add N(0, (delta_f * sigma_dp)^2) noise to every class hypervector. Eq. (8)."""
    std = delta_f * dp_sigma(epsilon, delta)
    noise = torch.normal(
        mean=0.0,
        std=std,
        size=class_hvs.shape,
        device=class_hvs.device,
        dtype=class_hvs.dtype,
        generator=generator,
    )
    return class_hvs + noise


def snr(
    n_train: int,
    mu_c: float,
    hv_dim: int,
    epsilon: float,
    delta: float = 1e-4,
    n_retrain_epochs: int = 1,
) -> float:
    """Signal-to-noise ratio of the inference dot product. Eq. (13) and (14).

    Signal grows with the number of training samples ``n_train`` and their
    average within-class similarity ``mu_c``; noise grows with the hypervector
    dimension ``hv_dim`` and with the noise multiplier. Retraining for ``T``
    passes scales the signal by ``T`` and the noise by ``sqrt(T)``, hence the
    ``sqrt(T)`` factor.

    The three practical consequences, all visible in Figure 10: more data buys
    privacy for free, a smaller hyperspace raises SNR, and more retraining
    passes raise it further.
    """
    return (
        math.sqrt(n_retrain_epochs) * n_train * mu_c / (hv_dim * dp_sigma(epsilon, delta))
    )


def snr_db(*args, **kwargs) -> float:
    """SNR expressed in decibels (20 log10, amplitude convention)."""
    value = snr(*args, **kwargs)
    return 20 * math.log10(value) if value > 0 else float("-inf")


def epsilon_from_sigma(sigma_dp: float, delta: float = 1e-4) -> float:
    """Invert Eq. (7): the privacy budget implied by a noise multiplier."""
    return math.sqrt(2 * math.log(1.25 / delta)) / sigma_dp
