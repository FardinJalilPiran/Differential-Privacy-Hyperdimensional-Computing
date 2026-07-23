"""Training privacy: sweep the basis sigma to find sigma_b* at a given epsilon.

This is Algorithm 2 of the paper. For each candidate sigma_b we encode, train,
retrain, add DP noise to the class hypervectors, and measure accuracy on a held
out set. The sigma that maximises accuracy at the lowest acceptable epsilon is
sigma_b*.

Example
-------
    python scripts/run_training_privacy.py --epsilon 0.8 --hv-dim 1000
    python scripts/run_training_privacy.py --epsilon 0.8 --epsilon 2.0 --n-sigma 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dphd import (  # noqa: E402
    accuracy,
    encode,
    make_basis,
    mean_class_similarity,
    privatize,
    retrain,
    sensitivity,
    set_seed,
    get_device,
    train_class_hvs,
)
from dphd.data import load_dataset, split_to_tensors  # noqa: E402
from dphd.plotting import plot_sigma_sweep  # noqa: E402
from dphd.privacy import snr_db  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=["mnist", "synthetic"])
    p.add_argument("--data-root", default="data", help="where MNIST is cached")
    p.add_argument("--n-samples", type=int, default=5000, help="total samples before the 70/30 split")
    p.add_argument("--downsample", type=int, default=2, help="pooling factor on 28x28 images")
    p.add_argument("--hv-dim", type=int, default=1000, help="hypervector dimension D")
    p.add_argument("--n-epochs", type=int, default=4, help="retraining passes T")
    p.add_argument("--epsilon", type=float, action="append", default=None, help="privacy budget; repeat for several")
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--sigma-max", type=float, default=1.2)
    p.add_argument("--n-sigma", type=int, default=30)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--out-dir", default="results/training_privacy")
    args = p.parse_args()
    if args.epsilon is None:
        args.epsilon = [0.8]
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    load_kwargs = dict(seed=args.seed, n_samples=args.n_samples)
    if args.dataset == "mnist":
        load_kwargs.update(root=args.data_root, downsample=args.downsample)
    X, y = load_dataset(args.dataset, **load_kwargs)
    X_tr, X_te, y_tr, y_te = split_to_tensors(X, y, device=device, seed=args.seed)

    n_classes = int(torch.unique(y_tr).numel())
    n_features = X_tr.shape[1]
    print(f"{len(X_tr)} train / {len(X_te)} test, {n_features} features, {n_classes} classes")

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma)
    results = {eps: [] for eps in args.epsilon}
    snr_records = {eps: [] for eps in args.epsilon}

    for i, sigma_b in enumerate(sigmas, 1):
        basis, bias = make_basis(n_features, args.hv_dim, float(sigma_b), device)
        enc_tr = encode(X_tr, basis, bias)
        enc_te = encode(X_te, basis, bias)

        class_hvs = train_class_hvs(enc_tr, y_tr, n_classes)
        class_hvs = retrain(class_hvs, enc_tr, y_tr, n_epochs=args.n_epochs)

        delta_f = sensitivity(enc_tr)
        mu_c = mean_class_similarity(enc_tr, y_tr)

        line = [f"[{i:>3}/{len(sigmas)}] sigma_b={sigma_b:.3f}"]
        for eps in args.epsilon:
            noisy = privatize(class_hvs, epsilon=eps, delta_f=delta_f, delta=args.delta)
            acc = accuracy(enc_te, noisy, y_te)
            results[eps].append(acc)
            snr_records[eps].append(
                snr_db(len(enc_tr), mu_c, args.hv_dim, eps, args.delta, args.n_epochs)
            )
            line.append(f"eps={eps}: acc={acc * 100:5.2f}%")
        print("  ".join(line))

    summary = {}
    for eps in args.epsilon:
        acc_pct = np.asarray(results[eps]) * 100
        best = int(np.argmax(acc_pct))
        summary[str(eps)] = {
            "sigma_b_star": float(sigmas[best]),
            "best_accuracy_pct": float(acc_pct[best]),
            "predicted_snr_db": float(snr_records[eps][best]),
        }
        print(
            f"\nepsilon={eps}: sigma_b* = {sigmas[best]:.3f}, "
            f"accuracy = {acc_pct[best]:.2f}%, predicted SNR = {snr_records[eps][best]:.2f} dB"
        )
        plot_sigma_sweep(sigmas, acc_pct, eps, out_path=out_dir / f"sigma_sweep_eps{eps}.png")

    np.savez(
        out_dir / "sigma_sweep.npz",
        sigmas=sigmas,
        **{f"accuracy_eps{eps}": np.asarray(results[eps]) for eps in args.epsilon},
        **{f"snr_db_eps{eps}": np.asarray(snr_records[eps]) for eps in args.epsilon},
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved to {out_dir}/")


if __name__ == "__main__":
    main()
