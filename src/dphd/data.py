"""Dataset loading for DP-HD.

The experiments in the paper use in-situ high-speed camera data collected on an
EOS M270 LPBF machine at NIST. That dataset is proprietary and is not
redistributed here. To keep every result in this repository reproducible by
anyone, the public **MNIST** dataset is used as a stand-in. It has the same
shape of problem the framework expects: flattened grayscale images, a modest
number of classes, and enough samples that the effect of DP noise on accuracy
is visible.

Nothing in `dphd.encoding`, `dphd.model`, `dphd.privacy` or
`dphd.inference_privacy` is dataset specific. To run on your own data, add a
branch to :func:`load_dataset` that returns:

    X : float32 array of shape (n_samples, n_features), values roughly in [0, 1]
    y : int64 array of shape (n_samples,), labels in 0..n_classes-1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

_MNIST_SIDE = 28


def load_mnist(
    root: str | Path = "data",
    n_samples: int | None = 5000,
    downsample: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Download (once) and load MNIST as flat feature vectors in [0, 1].

    Parameters
    ----------
    root : path
        Where torchvision caches the raw files. Created if missing.
    n_samples : int or None
        Take a stratified subsample of this size. ``None`` uses all 70,000
        images. The default of 5000 keeps a full sigma sweep to a few minutes.
    downsample : int
        Average-pool factor applied to the 28x28 image before flattening. A
        value of 2 gives 14x14 = 196 features. Mirrors the 8x8 block reduction
        applied to the 256x256 AM frames in the paper.
    seed : int
        Controls the stratified subsample.
    """
    from torchvision import datasets  # imported lazily so the rest works without it

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train = datasets.MNIST(root=str(root), train=True, download=True)
    test = datasets.MNIST(root=str(root), train=False, download=True)

    images = torch.cat([train.data, test.data], dim=0).float() / 255.0  # (70000, 28, 28)
    labels = torch.cat([train.targets, test.targets], dim=0).long()

    if downsample > 1:
        if _MNIST_SIDE % downsample != 0:
            raise ValueError(f"downsample must divide {_MNIST_SIDE}, got {downsample}")
        images = torch.nn.functional.avg_pool2d(
            images.unsqueeze(1), kernel_size=downsample
        ).squeeze(1)

    X = images.reshape(images.shape[0], -1).numpy().astype(np.float32)
    y = labels.numpy().astype(np.int64)

    if n_samples is not None and n_samples < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, random_state=seed, stratify=y
        )

    return X, y


def load_synthetic(
    n_samples: int = 5000,
    n_features: int = 196,
    n_classes: int = 8,
    class_gap: float = 5.0,
    sample_spread: float = 1.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blobs, one per class, as used for Figures 5 and 6 of the paper.

    ``class_gap`` is the separation between class centres (dC) and
    ``sample_spread`` the within-class standard deviation (sigma_sample). These
    are the two quantities the paper shows the optimal basis sigma depends on.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_classes, n_features))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    centres *= class_gap

    y = rng.integers(0, n_classes, size=n_samples)
    X = centres[y] + rng.normal(scale=sample_spread, size=(n_samples, n_features))
    return X.astype(np.float32), y.astype(np.int64)


def load_dataset(name: str = "mnist", **kwargs):
    """Dispatch to a dataset loader by name."""
    loaders = {"mnist": load_mnist, "synthetic": load_synthetic}
    if name not in loaders:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(loaders)}")
    return loaders[name](**kwargs)


def split_to_tensors(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    test_size: float = 0.3,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified train/test split, shuffled, returned as tensors on ``device``."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    rng = np.random.default_rng(seed)
    tr_perm, te_perm = rng.permutation(len(X_tr)), rng.permutation(len(X_te))
    X_tr, y_tr = X_tr[tr_perm], y_tr[tr_perm]
    X_te, y_te = X_te[te_perm], y_te[te_perm]

    return (
        torch.as_tensor(X_tr, dtype=dtype, device=device),
        torch.as_tensor(X_te, dtype=dtype, device=device),
        torch.as_tensor(y_tr, dtype=torch.long, device=device),
        torch.as_tensor(y_te, dtype=torch.long, device=device),
    )
