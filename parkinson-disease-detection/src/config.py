"""Central configuration: paths, hyperparameters, seeds and helpers.

Every tunable value lives here so experiments are reproducible and the other
modules stay free of magic numbers.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"                 # Roboflow export (git-ignored)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"               # .keras files (git-ignored)
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"      # best-weights checkpoints
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

# --------------------------------------------------------------------------- #
# Roboflow
# --------------------------------------------------------------------------- #
ROBOFLOW_WORKSPACE = "anosh-goodwin"
ROBOFLOW_PROJECT = "parkinson-disease-detection-1eeyh"
ROBOFLOW_VERSION = 1            # change if the project has newer dataset versions
ROBOFLOW_FORMAT = "folder"      # classification export: train/ valid/ test/
ROBOFLOW_API_KEY_PLACEHOLDER = "YOUR_ROBOFLOW_API_KEY"

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
SPLITS = ("train", "valid", "test")
SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val", "validation"),
    "test": ("test",),
}
IMG_SIZE = (224, 224)
IMG_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 16

NEGATIVE_CLASS = "healthy"      # label 0
POSITIVE_CLASS = "parkinson"    # label 1 (the class we care about detecting)
CLASS_NAMES = (NEGATIVE_CLASS, POSITIVE_CLASS)
POSITIVE_ALIASES = {"parkinson", "parkinsons", "parkinson's", "parkinson_disease", "pd"}
NEGATIVE_ALIASES = {"healthy", "normal", "control"}

# Leakage / duplicate detection (perceptual hash)
PHASH_SIZE = 16                 # 16x16 -> 256-bit hash
PHASH_THRESHOLD = 8             # max Hamming distance to call two images near-duplicates

# Augmentation (training split only). Disable flips if the images are orientation
# sensitive (e.g. handwriting / spiral direction).
AUG_HORIZONTAL_FLIP = True
AUG_ROTATION = 0.08             # fraction of 2*pi
AUG_ZOOM = 0.10
AUG_TRANSLATION = 0.05
AUG_CONTRAST = 0.15
AUG_BRIGHTNESS = 0.10

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
SEED = 42
MODEL_NAMES = ("custom_cnn", "vgg16", "resnet50")

EPOCHS_SCRATCH = 60             # custom CNN
EPOCHS_HEAD = 30                # transfer learning phase 1 (frozen backbone)
EPOCHS_FINE_TUNE = 30           # transfer learning phase 2 (partial unfreeze)

LR_SCRATCH = 1e-3
LR_HEAD = 1e-3
LR_FINE_TUNE = 1e-5

EARLY_STOP_PATIENCE = 8
EARLY_STOP_PATIENCE_SCRATCH = 12
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.3
MIN_LR = 1e-7

DROPOUT = 0.5
L2_REG = 1e-4
UNFREEZE_LAYERS = {"vgg16": 4, "resnet50": 30}   # trailing layers unfrozen in phase 2

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
DEFAULT_THRESHOLD = 0.5
N_BOOTSTRAP = 1000              # bootstrap resamples for confidence intervals


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    """Create all output directories."""
    # DATA_DIR is deliberately NOT created here: Roboflow must create it itself.
    for d in (MODELS_DIR, CHECKPOINT_DIR, PLOTS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - TF not installed yet
        pass


def get_api_key() -> str:
    """Resolve the Roboflow API key without ever hard-coding it.

    Order: ``ROBOFLOW_API_KEY`` env var -> Colab Secrets (``userdata``).
    Raises if only the placeholder is available.
    """
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        try:
            from google.colab import userdata  # type: ignore

            key = userdata.get("ROBOFLOW_API_KEY")
        except Exception:
            key = None
    if not key or key == ROBOFLOW_API_KEY_PLACEHOLDER:
        raise RuntimeError(
            "Roboflow API key not found. Set the ROBOFLOW_API_KEY environment variable "
            "or add it to Colab Secrets (key icon in the left sidebar) as 'ROBOFLOW_API_KEY'."
        )
    return key
