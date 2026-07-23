"""Inference privacy: drop low-variance dimensions and measure both sides.

Two quantities are tracked as more dimensions are zeroed: the accuracy of the
HD classifier, and the NMSE of an adversarial decoder trying to reconstruct the
input image from the query hypervector. Dropping low-variance dimensions should
leave the first roughly flat while pushing the second up. Dropping high-variance
dimensions is run as a control.

Example
-------
    python scripts/run_inference_privacy.py --n-runs 5
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
    drop_dimensions,
    encode,
    make_basis,
    retrain,
    set_seed,
    get_device,
    train_class_hvs,
    variance_order,
)
from dphd.data import load_dataset, split_to_tensors  # noqa: E402
from dphd.decoder import nmse, train_decoder  # noqa: E402
from dphd.plotting import plot_accuracy_vs_nmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=["mnist", "synthetic"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--hv-dim", type=int, default=1000)
    p.add_argument("--sigma-b", type=float, default=0.2, help="use sigma_b* from run_training_privacy.py")
    p.add_argument("--n-epochs", type=int, default=4, help="HD retraining passes")
    p.add_argument("--max-drop", type=int, default=60, help="maximum %% of dimensions dropped")
    p.add_argument("--drop-step", type=int, default=5)
    p.add_argument("--n-runs", type=int, default=5, help="repetitions averaged over (paper uses 100)")
    p.add_argument("--decoder-epochs", type=int, default=10)
    p.add_argument("--decoder-hidden", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--out-dir", default="results/inference_privacy")
    return p.parse_args()


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

    fractions = np.arange(0, args.max_drop + 1, args.drop_step) / 100.0
    acc_low, acc_high, err_low, err_high = [], [], [], []

    for run in range(args.n_runs):
        X_tr, X_te, y_tr, y_te = split_to_tensors(X, y, device=device, seed=args.seed + run)
        n_classes = int(torch.unique(y_tr).numel())
        n_features = X_tr.shape[1]

        basis, bias = make_basis(n_features, args.hv_dim, args.sigma_b, device)
        enc_tr = encode(X_tr, basis, bias)
        enc_te = encode(X_te, basis, bias)

        class_hvs = train_class_hvs(enc_tr, y_tr, n_classes)
        class_hvs = retrain(class_hvs, enc_tr, y_tr, n_epochs=args.n_epochs)

        print(f"[run {run + 1}/{args.n_runs}] training the adversarial decoder ...")
        decoder = train_decoder(
            enc_tr, X_tr, hidden_size=args.decoder_hidden, epochs=args.decoder_epochs
        )

        order = variance_order(enc_te)
        run_acc_low, run_acc_high, run_err_low, run_err_high = [], [], [], []

        for frac in fractions:
            q_low = drop_dimensions(enc_te, order, frac, mode="low")
            q_high = drop_dimensions(enc_te, order, frac, mode="high")

            run_acc_low.append(accuracy(q_low, class_hvs, y_te))
            run_acc_high.append(accuracy(q_high, class_hvs, y_te))
            with torch.no_grad():
                run_err_low.append(nmse(decoder(q_low), X_te))
                run_err_high.append(nmse(decoder(q_high), X_te))

        acc_low.append(run_acc_low)
        acc_high.append(run_acc_high)
        err_low.append(run_err_low)
        err_high.append(run_err_high)
        print(
            f"           accuracy {run_acc_low[0] * 100:.2f}% -> {run_acc_low[-1] * 100:.2f}%, "
            f"NMSE {run_err_low[0]:.3f} -> {run_err_low[-1]:.3f}"
        )

    acc_low = np.mean(acc_low, axis=0) * 100
    acc_high = np.mean(acc_high, axis=0) * 100
    err_low = np.mean(err_low, axis=0)
    err_high = np.mean(err_high, axis=0)
    drop_pct = fractions * 100

    summary = {
        "max_drop_pct": float(drop_pct[-1]),
        "accuracy_drop_pct_low_variance": float(acc_low[0] - acc_low[-1]),
        "accuracy_drop_pct_high_variance": float(acc_high[0] - acc_high[-1]),
        "nmse_increase_pct_low_variance": float((err_low[-1] / err_low[0] - 1) * 100),
        "nmse_increase_pct_high_variance": float((err_high[-1] / err_high[0] - 1) * 100),
    }
    print("\n" + json.dumps(summary, indent=2))

    np.savez(
        out_dir / "inference_privacy.npz",
        drop_pct=drop_pct,
        accuracy_low=acc_low,
        accuracy_high=acc_high,
        nmse_low=err_low,
        nmse_high=err_high,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_accuracy_vs_nmse(
        drop_pct, acc_low, acc_high, err_low, err_high, out_path=out_dir / "accuracy_vs_nmse.png"
    )
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
