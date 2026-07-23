"""Inference privacy: dropping low-variance dimensions (Section 3.4.2).

A query hypervector sent to the model can be inverted by an attacker to recover
the raw image. Not every dimension carries class information, though: some are
nearly constant across all queries and encode only what is common to the
process. Zeroing those costs the classifier almost nothing while removing
information the attacker's decoder needs.

Dropping *high*-variance dimensions instead is the control condition: it
degrades reconstruction by a similar amount but wrecks accuracy, which is what
shows the choice of which dimensions to drop is doing real work.
"""

from __future__ import annotations

import torch


def variance_order(queries: torch.Tensor) -> torch.Tensor:
    """Indices of the hypervector dimensions sorted by ascending variance."""
    return torch.argsort(torch.var(queries, dim=0))


def drop_dimensions(
    queries: torch.Tensor,
    order: torch.Tensor,
    fraction: float,
    mode: str = "low",
) -> torch.Tensor:
    """Zero out a fraction of dimensions, chosen from either end of ``order``.

    Parameters
    ----------
    fraction : float in [0, 1]
        Share of the hypervector dimensions to suppress.
    mode : {"low", "high"}
        "low" drops the least informative dimensions (the privacy mechanism);
        "high" drops the most informative ones (the control).
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    if mode not in {"low", "high"}:
        raise ValueError("mode must be 'low' or 'high'")

    out = queries.clone()
    k = int(fraction * queries.shape[1])
    if k == 0:
        return out
    idx = order[:k] if mode == "low" else order[-k:]
    out[:, idx] = 0.0
    return out
