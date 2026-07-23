"""SNR analysis: predict the accuracy cost of DP noise before paying it.

This is the explainability contribution of the paper. The SNR of Eq. (13)-(14)
is computed analytically from quantities you know before adding any noise (the
training set size, the within-class similarity mu_c, the hypervector dimension,
and the privacy budget). This script computes it and, optionally, measures the
accuracy actually obtained so you can see how well the prediction tracks.

Example
-------
    python scripts/run_snr_analysis.py --validate
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
from dphd.plotting import plot_snr  # noqa: E402
from dphd.privacy import epsilon_from_sigma, snr_db  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=["mnist", "synthetic"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--sigma-b", type=float, default=0.2)
    p.add_argument("--hv-dims", type=int, nargs="+", default=[1000, 3000, 10000])
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--validate", action="store_true", help="also measure accuracy at each epsilon")
    p.add_argument("--eps-grid", type=float, nargs="+", default=[0.6, 0.8, 1.0, 2.0, 3.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--out-dir", default="results/snr")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = dict(seed=args.seed, n_samples=args.n_samples)
    if args.dataset == "mnist":
        load_kwargs.update(root=args.data_root, downsample=args.downsample)
    X, y = load_dataset(args.dataset, **load_kwargs)
    X_tr, X_te, y_tr, y_te = split_to_tensors(X, y, device=device, seed=args.seed)
    n_classes = int(torch.unique(y_tr).numel())
    n_features = X_tr.shape[1]

    # --- SNR against the DP noise multiplier, one curve per hyperspace size ---
    sigma_dp_grid = np.linspace(0.01, 0.20, 20)
    curves, mu_by_dim = {}, {}
    for D in args.hv_dims:
        basis, bias = make_basis(n_features, D, args.sigma_b, device)
        enc_tr = encode(X_tr, basis, bias)
        mu_c = mean_class_similarity(enc_tr, y_tr)
        mu_by_dim[D] = mu_c
        curves[f"D = {D}"] = np.array(
            [
                snr_db(len(enc_tr), mu_c, D, epsilon_from_sigma(s, args.delta), args.delta, args.n_epochs)
                for s in sigma_dp_grid
            ]
        )
        print(f"D={D}: mu_c={mu_c:.4f}, SNR at sigma_dp=0.01 is {curves[f'D = {D}'][0]:.2f} dB")

    plot_snr(sigma_dp_grid, curves, r"$\sigma_{dp}$", out_path=out_dir / "snr_vs_noise.png")

    # --- SNR against retraining passes, one curve per training set size ---
    D = args.hv_dims[0]
    basis, bias = make_basis(n_features, D, args.sigma_b, device)
    enc_tr_full = encode(X_tr, basis, bias)
    mu_c = mean_class_similarity(enc_tr_full, y_tr)
    epochs_grid = np.arange(1, 11)
    size_curves = {}
    for n in [1000, 2000, min(5000, len(X_tr))]:
        if n > len(X_tr):
            continue
        size_curves[f"N = {n}"] = np.array(
            [snr_db(n, mu_c, D, 1.0, args.delta, int(t)) for t in epochs_grid]
        )
    plot_snr(epochs_grid, size_curves, "Retraining epochs", out_path=out_dir / "snr_vs_epochs.png")

    report = {
        "mu_c_by_hv_dim": {str(k): float(v) for k, v in mu_by_dim.items()},
        "n_train": int(len(X_tr)),
    }

    # --- optional: does predicted SNR actually track measured accuracy? ---
    if args.validate:
        D = args.hv_dims[0]
        basis, bias = make_basis(n_features, D, args.sigma_b, device)
        enc_tr = encode(X_tr, basis, bias)
        enc_te = encode(X_te, basis, bias)
        class_hvs = train_class_hvs(enc_tr, y_tr, n_classes)
        class_hvs = retrain(class_hvs, enc_tr, y_tr, n_epochs=args.n_epochs)
        delta_f = sensitivity(enc_tr)
        mu_c = mean_class_similarity(enc_tr, y_tr)

        table = []
        print(f"\n{'epsilon':>8} {'SNR (dB)':>10} {'accuracy':>10}")
        for eps in args.eps_grid:
            noisy = privatize(class_hvs, epsilon=eps, delta_f=delta_f, delta=args.delta)
            acc = accuracy(enc_te, noisy, y_te) * 100
            s = snr_db(len(enc_tr), mu_c, D, eps, args.delta, args.n_epochs)
            table.append({"epsilon": eps, "snr_db": s, "accuracy_pct": acc})
            print(f"{eps:>8.2f} {s:>10.2f} {acc:>9.2f}%")
        report["validation"] = table

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved to {out_dir}/")


if __name__ == "__main__":
    main()
