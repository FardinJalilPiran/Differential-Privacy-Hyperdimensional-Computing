"""Figure helpers. Each function mirrors one figure from the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out_path: str | Path | None):
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_sigma_sweep(
    sigma_b: np.ndarray,
    accuracy: np.ndarray,
    epsilon: float,
    threshold: float = 90.0,
    out_path: str | Path | None = None,
):
    """Accuracy against basis sigma at one privacy budget (Figure 8).

    Shades the interval where accuracy clears ``threshold``; the width of that
    interval is the tolerance you have when picking sigma_b.
    """
    accuracy = np.asarray(accuracy, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.fill_between(sigma_b, accuracy, color="#7A4FA3", alpha=0.55)

    above = accuracy >= threshold
    if above.any():
        lo = sigma_b[np.argmax(above)]
        hi = sigma_b[len(sigma_b) - np.argmax(above[::-1]) - 1]
        ax.fill_between(
            sigma_b,
            0,
            accuracy,
            where=(sigma_b >= lo) & (sigma_b <= hi),
            facecolor="none",
            hatch="///",
            edgecolor="black",
            linewidth=0.0,
        )
        ax.annotate(
            rf"$\sigma_b$ = {lo:.2f}",
            xy=(lo, 1),
            xytext=(lo, 22),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
            ha="center",
        )
        ax.annotate(
            rf"$\sigma_b$ = {hi:.2f}",
            xy=(hi, 1),
            xytext=(hi, 22),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
            ha="center",
        )

    best = int(np.argmax(accuracy))
    ax.annotate(
        f"max accuracy = {accuracy[best]:.1f}%\n" + rf"at $\sigma_b^*$ = {sigma_b[best]:.2f}",
        xy=(sigma_b[best], accuracy[best]),
        xytext=(sigma_b[best] + 0.1, min(accuracy[best] + 12, 98)),
        arrowprops=dict(facecolor="#7A4FA3", shrink=0.05, width=1, headwidth=6),
    )

    ax.set_xlabel(r"$\sigma_b$ (basis standard deviation)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(rf"Privacy budget $\epsilon$ = {epsilon}", fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_xlim(sigma_b.min(), sigma_b.max())
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    return _save(fig, out_path)


def plot_accuracy_vs_nmse(
    drop_pct: np.ndarray,
    acc_low: np.ndarray,
    acc_high: np.ndarray,
    nmse_low: np.ndarray,
    nmse_high: np.ndarray,
    out_path: str | Path | None = None,
):
    """Twin-axis accuracy / reconstruction error against dropped dimensions (Figure 13).

    The useful region is where the cyan curves stay flat while the magenta ones
    climb: the classifier is unaffected, the attacker is not.
    """
    acc_c, nmse_c = "#0DBDD3", "#DD3394"
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.grid(color="gray", linestyle="--", linewidth=1.2, alpha=0.2)

    ax.plot(drop_pct, acc_low, "-o", lw=1.6, color=acc_c, label="Low variance – accuracy")
    ax.plot(drop_pct, acc_high, "-^", lw=1.6, color=acc_c, ls="--", label="High variance – accuracy")
    ax.set_ylabel("Accuracy (%)", fontsize=13, color=acc_c)
    ax.tick_params(axis="y", colors=acc_c)
    ax.set_xlabel("Dropped dimensions (%)", fontsize=13)

    ax2 = ax.twinx()
    ax2.plot(drop_pct, nmse_low, "-o", lw=1.6, color=nmse_c, label="Low variance – NMSE")
    ax2.plot(drop_pct, nmse_high, "-^", lw=1.6, color=nmse_c, ls="--", label="High variance – NMSE")
    ax2.set_ylabel("NMSE", fontsize=13, color=nmse_c)
    ax2.tick_params(axis="y", colors=nmse_c)

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    return _save(fig, out_path)


def plot_snr(
    x: np.ndarray,
    curves: dict[str, np.ndarray],
    xlabel: str,
    out_path: str | Path | None = None,
):
    """SNR in dB against a swept quantity (Figure 10)."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.grid(color="gray", linestyle="--", linewidth=1.2, alpha=0.2)
    for label, y in curves.items():
        ax.plot(x, y, "-o", lw=1.6, ms=4, label=label)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("SNR (dB)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)
