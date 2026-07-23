"""DP-HD: Differential Privacy - Hyperdimensional Computing.

Reference implementation for:
    Piran, Poduval, Barkam, Imani & Imani (2025),
    "Explainable differential privacy-hyperdimensional computing for balancing
    privacy and transparency in additive manufacturing monitoring",
    Engineering Applications of Artificial Intelligence, 147, 110282.
"""

__version__ = "1.0.0"

from .data import load_dataset
from .encoding import make_basis, encode
from .model import train_class_hvs, retrain, predict, accuracy, mean_class_similarity
from .privacy import dp_sigma, sensitivity, privatize, snr, snr_db
from .decoder import DecoderNetwork, train_decoder, nmse
from .inference_privacy import variance_order, drop_dimensions
from .utils import set_seed, get_device

__all__ = [
    "load_dataset",
    "make_basis",
    "encode",
    "train_class_hvs",
    "retrain",
    "predict",
    "accuracy",
    "mean_class_similarity",
    "dp_sigma",
    "sensitivity",
    "privatize",
    "snr",
    "snr_db",
    "DecoderNetwork",
    "train_decoder",
    "nmse",
    "variance_order",
    "drop_dimensions",
    "set_seed",
    "get_device",
]
