"""The attacker's decoder (Sections 3.4.1 and 5.2).

Direct inversion of the cosine encoder is infeasible once the basis spread is
large, so a realistic adversary trains a regression network that maps
hypervectors back to feature vectors. This module implements that adversary, so
that the inference-privacy mechanism can be evaluated against it rather than
against an assumption.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderNetwork(nn.Module):
    """Three-layer MLP: hypervector -> feature vector."""

    def __init__(self, hv_dim: int, n_features: int, hidden_size: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(hv_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_decoder(
    encoded: torch.Tensor,
    targets: torch.Tensor,
    hidden_size: int = 512,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = False,
) -> DecoderNetwork:
    """Fit the adversary on (hypervector, feature vector) pairs."""
    model = DecoderNetwork(
        hv_dim=encoded.shape[1], n_features=targets.shape[1], hidden_size=hidden_size
    ).to(device=encoded.device, dtype=encoded.dtype)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running = 0.0
        for i in range(0, len(encoded), batch_size):
            xb, yb = encoded[i : i + batch_size], targets[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        if verbose:
            print(f"  decoder epoch {epoch + 1}/{epochs}  mse={running / len(encoded):.5f}")

    return model


@torch.no_grad()
def nmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Normalised mean square error: MSE divided by the variance of the target.

    Higher means the reconstruction is worse, i.e. the defence is working.
    """
    mse = ((predictions - targets) ** 2).mean().item()
    return mse / targets.var().item()


@torch.no_grad()
def psnr(predictions: torch.Tensor, targets: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB, as reported in Figure 12."""
    import math

    mse = ((predictions - targets) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(data_range) - 10 * math.log10(mse)
