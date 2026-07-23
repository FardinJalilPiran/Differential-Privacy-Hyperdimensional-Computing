"""Encoding: feature vectors -> hypervectors (Section 3.1.1 of the paper).

Each element of a hypervector is ``h_i = cos(F . B_i + u_i)`` where ``B_i`` is a
Gaussian random basis vector with standard deviation ``sigma_b`` and ``u_i`` is
drawn uniformly from ``[0, 2*pi)``. This is the Random Fourier Feature map, so
``sigma_b`` sets the bandwidth of the approximated kernel:

* small ``sigma_b`` -> *inclusive* encoding, hypervectors stay similar even for
  distant inputs (good for generalising over similar features);
* large ``sigma_b`` -> *exclusive* encoding, similarity tracks true distance
  (good for separating fine-grained patterns).

Choosing ``sigma_b`` is what Algorithm 2 in the paper searches over.
"""

from __future__ import annotations

import torch


def make_basis(
    n_features: int,
    hv_dim: int,
    sigma_b: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    with_bias: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample the random basis matrix and the uniform phase offsets.

    Returns
    -------
    basis : (n_features, hv_dim) tensor, entries ~ N(0, sigma_b^2)
    bias : (hv_dim,) tensor of phases ~ U(0, 2*pi), or None if ``with_bias`` is
        False. The decoding experiment of Section 3.4.1 assumes no phase
        offset, so it passes ``with_bias=False``.
    """
    basis = torch.normal(
        mean=0.0,
        std=sigma_b,
        size=(n_features, hv_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    bias = None
    if with_bias:
        bias = (
            torch.rand(hv_dim, device=device, dtype=dtype, generator=generator)
            * 2
            * torch.pi
        )
    return basis, bias


def encode(
    X: torch.Tensor, basis: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Project ``X`` (n_samples, n_features) into (n_samples, hv_dim)."""
    projected = X @ basis
    if bias is not None:
        projected = projected + bias
    return torch.cos(projected)
