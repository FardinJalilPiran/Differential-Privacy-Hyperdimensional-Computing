# Differential Privacy–Hyperdimensional Computing (DP-HD)

Reference implementation of **DP-HD**, a framework that combines Explainable AI with
vector symbolic paradigms to *quantify and predict* the effect of Differential Privacy (DP)
noise on model accuracy, using a Signal-to-Noise Ratio (SNR) metric.

Because the SNR can be computed from quantities known before any noise is added — the
training set size, the within-class similarity of the encoded data, the hypervector
dimension, and the privacy budget — DP-HD lets you choose a privacy level and know the
accuracy cost in advance, rather than discovering it after training.

> Piran, F. J., Poduval, P. P., Barkam, H. E., Imani, M., & Imani, F. (2025).
> *Explainable differential privacy-hyperdimensional computing for balancing privacy
> and transparency in additive manufacturing monitoring.*
> Engineering Applications of Artificial Intelligence, 147, 110282.
> [doi:10.1016/j.engappai.2025.110282](https://doi.org/10.1016/j.engappai.2025.110282)

---

## A note on the data

The results in the paper come from in-situ high-speed camera measurements of a laser
powder bed fusion build (EOS M270 at NIST). **That dataset is proprietary and is not
included here.** So that every experiment in this repository is reproducible by anyone,
the code ships with **MNIST**, which torchvision downloads automatically on first run.

MNIST is a stand-in, not a replication: absolute numbers will differ from the paper. The
qualitative behaviour it is meant to demonstrate — that accuracy peaks at an
intermediate basis sigma, that low-variance dimensions can be discarded almost for free,
that the SNR tracks measured accuracy — reproduces.

Swapping in your own data takes one function; see [`data/README.md`](data/README.md).

---

## Quickstart

```bash
git clone https://github.com/FardinJalilPiran/Differential-Privacy-Hyperdimensional-Computing.git
cd Differential-Privacy-Hyperdimensional-Computing

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. find the optimal basis sigma at a given privacy budget
python scripts/run_training_privacy.py --epsilon 0.8

# 2. evaluate the inference-privacy mechanism against an adversarial decoder
python scripts/run_inference_privacy.py --sigma-b 0.2 --n-runs 5

# 3. the SNR / explainability analysis
python scripts/run_snr_analysis.py --validate
```

MNIST downloads to `data/` on first run (~55 MB). Figures, arrays, and a JSON summary
are written to `results/`. A CUDA GPU is used automatically when present; everything
also runs on CPU, and on Apple Silicon via MPS.

---

## Repository layout

```
.
├── src/dphd/                      # the library — nothing here is dataset specific
│   ├── data.py                    # MNIST + synthetic loaders, train/test split
│   ├── encoding.py                # random Fourier feature encoder      §3.1.1
│   ├── model.py                   # bundling, retraining, inference      §3.1.2–3.1.4
│   ├── privacy.py                 # Gaussian mechanism, sensitivity, SNR §3.2–3.3.1
│   ├── inference_privacy.py       # variance-based dimension dropping    §3.4.2
│   ├── decoder.py                 # the attacker's decoder, NMSE, PSNR   §3.4.1
│   ├── plotting.py                # figure helpers
│   └── utils.py                   # seeding, device selection
├── scripts/                       # command-line experiments
│   ├── run_training_privacy.py    # Algorithm 2 — sigma sweep            Fig. 8
│   ├── run_inference_privacy.py   # accuracy vs reconstruction error     Fig. 13
│   └── run_snr_analysis.py        # SNR vs noise, dimension, epochs      Fig. 10
├── notebooks/                     # the same two experiments, interactively
│   ├── 01_training_privacy.ipynb
│   └── 02_inference_privacy.ipynb
├── data/                          # datasets land here (gitignored)
├── results/                       # figures and arrays land here (gitignored)
├── requirements.txt
└── pyproject.toml                 # optional: pip install -e .
```

---

## How the method works

**Encoding.** A feature vector `F` becomes a hypervector via `h_i = cos(F · B_i + u_i)`,
with `B_i ~ N(0, sigma_b²)` and `u_i ~ U(0, 2π)`. This is a Random Fourier Feature map,
so `sigma_b` sets the kernel bandwidth: small values give an *inclusive* encoding where
distant inputs still look similar, large values an *exclusive* one where similarity
tracks true distance. Picking it well is what Algorithm 2 does.

**Training.** Hypervectors of the same class are summed into one class hypervector, then
refined by error-driven retraining: on a misprediction the sample is added to its true
class and subtracted from the predicted one.

**Privacy.** Gaussian noise `N(0, Δg²σ_dp²)` is added to the class hypervectors, with
`σ_dp = √(2 ln(1.25/δ)) / ε` and sensitivity `Δg` the largest L2 norm over encoded
training samples. What leaves the training environment is already private.

**Explainability.** At inference the score is a dot product, so signal and noise are both
Gaussian and their ratio is available in closed form:

```
SNR = √T · N · μ_c / (D · σ_dp)
```

with `N` training samples, `μ_c` the mean within-class similarity, `D` the hypervector
dimension, and `T` retraining passes. Three things follow directly, and all three are
worth knowing before you train: more data buys privacy nearly for free, a smaller
hyperspace raises SNR, and extra retraining passes raise it as `√T`.

**Inference privacy.** Query hypervectors can be inverted by an attacker. Dimensions with
low variance across queries carry little class information, so zeroing them costs the
classifier little while removing what the attacker's decoder relies on. Dropping
*high*-variance dimensions is the control: similar damage to the attacker, severe damage
to accuracy.

---

## Key parameters

| Flag | Symbol | Default | Notes |
|---|---|---|---|
| `--hv-dim` | `D` | 1000 | The only hyperparameter needing manual choice. Above the threshold that captures the features, larger buys little. |
| `--sigma-b` | `σ_b` | 0.2 | Basis spread. Depends only on the data distribution — not on `D`, not on training set size. Find it with `run_training_privacy.py`. |
| `--epsilon` | `ε` | 0.8 | Privacy budget. Lower is more private, and costs accuracy. |
| `--delta` | `δ` | 1e-4 | Should be below the inverse of the dataset size. |
| `--n-epochs` | `T` | 4 | Retraining passes. |
| `--max-drop` | — | 60 | Percentage of query dimensions zeroed for inference privacy. |

---

## Citation

```bibtex
@article{piran2025explainable,
  title   = {Explainable differential privacy-hyperdimensional computing for balancing
             privacy and transparency in additive manufacturing monitoring},
  author  = {Piran, Fardin Jalil and Poduval, Prathyush P and Barkam, Hamza Errahmouni
             and Imani, Mohsen and Imani, Farhad},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {147},
  pages   = {110282},
  year    = {2025},
  publisher = {Elsevier}
}
```

## Acknowledgment

Supported by the National Science Foundation (2127780, 2312517, 2434519); the
Semiconductor Research Corporation; the Office of Naval Research (N00014-21-1-2225,
N00014-22-1-2067); the Air Force Office of Scientific Research (FA9550-22-1-0253); UConn
Startup Funding; and gifts from Xilinx and Cisco. The authors thank NIST, and Dr. Brandon
Lane in particular, for providing data for this research.

## License

MIT — see [LICENSE](LICENSE).
