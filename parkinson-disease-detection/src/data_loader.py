"""Dataset download, EDA, leakage / duplicate checks and the tf.data pipeline."""
from __future__ import annotations

import hashlib
import itertools
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from . import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def _find_split_dir(root: Path, split: str) -> Optional[Path]:
    for name in config.SPLIT_ALIASES[split]:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def download_dataset(
    api_key: Optional[str] = None,
    version: int = config.ROBOFLOW_VERSION,
    location: Path = config.DATA_DIR,
    force: bool = False,
) -> Path:
    """Download the Roboflow export in ``folder`` classification format.

    Skips the download if all splits already exist (unless ``force``).
    """
    location = Path(location)
    if not force and all(_find_split_dir(location, s) for s in config.SPLITS):
        print(f"[data] Dataset already present at {location}")
        return location
    if force and location.exists():
        shutil.rmtree(location)
    if location.exists() and not any(location.iterdir()):
        location.rmdir()  # an empty pre-existing folder makes the Roboflow extract unreliable

    from roboflow import Roboflow  # imported lazily: only needed for download

    rf = Roboflow(api_key=api_key or config.get_api_key())
    project = rf.workspace(config.ROBOFLOW_WORKSPACE).project(config.ROBOFLOW_PROJECT)
    dataset = project.version(version).download(config.ROBOFLOW_FORMAT, location=str(location))
    root = Path(dataset.location)
    print(f"[data] Downloaded to {root}")

    # Verify the export really contains the splits (also handles one level of nesting).
    for candidate in [root, *sorted(p for p in root.glob("*") if p.is_dir())]:
        if all(_find_split_dir(candidate, s) for s in config.SPLITS):
            return candidate
    found = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir())[:20]
    raise FileNotFoundError(
        f"Download finished but no train/valid/test folders were found in {root}. "
        f"Folders present: {found}. Check ROBOFLOW_WORKSPACE / ROBOFLOW_PROJECT / ROBOFLOW_VERSION."
    )


# --------------------------------------------------------------------------- #
# 2. Manifest
# --------------------------------------------------------------------------- #
def class_to_label(class_dir_name: str) -> int:
    name = class_dir_name.strip().lower()
    if name in config.POSITIVE_ALIASES:
        return 1
    if name in config.NEGATIVE_ALIASES:
        return 0
    raise ValueError(
        f"Unrecognised class folder '{class_dir_name}'. Expected one of "
        f"{sorted(config.NEGATIVE_ALIASES | config.POSITIVE_ALIASES)}; "
        "adjust the aliases in src/config.py."
    )


def _source_id(path: Path) -> str:
    """Roboflow names files ``<original>.rf.<hash>.jpg`` - recover the original name."""
    return path.name.split(".rf.")[0]


def build_manifest(root: Path) -> pd.DataFrame:
    """Scan the export and return one row per image."""
    root = Path(root)
    rows = []
    for split in config.SPLITS:
        split_dir = _find_split_dir(root, split)
        if split_dir is None:
            raise FileNotFoundError(f"Split '{split}' not found under {root}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = class_to_label(class_dir.name)
            for f in sorted(class_dir.rglob("*")):
                if f.suffix.lower() in IMG_EXTS:
                    rows.append(
                        dict(
                            path=str(f),
                            split=split,
                            class_name=config.CLASS_NAMES[label],
                            label=label,
                            source_id=_source_id(f),
                        )
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root}")
    return df


# --------------------------------------------------------------------------- #
# 3. EDA
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, show: bool = True) -> pd.DataFrame:
    """Class balance, image-size statistics and a sample grid. Returns df with sizes."""
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    counts = counts.reindex(list(config.SPLITS))
    print("\n[eda] Images per split / class\n", counts, sep="")
    print(f"[eda] Total images: {len(df)}")

    train_counts = counts.loc["train"]
    ratio = train_counts.max() / max(train_counts.min(), 1)
    print(f"[eda] Train imbalance ratio (majority/minority): {ratio:.2f}")

    sizes = []
    for p in df.path:
        with Image.open(p) as im:
            sizes.append((im.width, im.height, im.mode))
    df = df.copy()
    df[["width", "height", "mode"]] = pd.DataFrame(sizes, index=df.index)
    print("\n[eda] Image size statistics\n", df[["width", "height"]].describe().round(1), sep="")
    print("[eda] Colour modes:", df["mode"].value_counts().to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_title("Images per split and class")
    axes[0].set_ylabel("count")
    sns.scatterplot(data=df, x="width", y="height", hue="class_name", ax=axes[1])
    axes[1].set_title("Original image dimensions")
    plt.tight_layout()
    fig.savefig(config.PLOTS_DIR / "eda_class_distribution.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # Sample grid (train)
    train = df[df.split == "train"]
    n = 4
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for r, cls in enumerate(config.CLASS_NAMES):
        sample = train[train.class_name == cls].sample(
            min(n, (train.class_name == cls).sum()), random_state=config.SEED
        )
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(sample):
                ax.imshow(Image.open(sample.iloc[c].path).convert("RGB"))
                ax.set_title(cls, fontsize=10)
    plt.tight_layout()
    fig.savefig(config.PLOTS_DIR / "eda_samples.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return df


# --------------------------------------------------------------------------- #
# 4. Leakage / duplicate detection
# --------------------------------------------------------------------------- #
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate_pairs(df: pd.DataFrame, threshold: int = config.PHASH_THRESHOLD) -> pd.DataFrame:
    """Find exact and near-duplicate image pairs (within and across splits).

    A pair is reported when any of the following holds:
      * byte-identical files (MD5)
      * perceptual-hash Hamming distance <= ``threshold``
      * same Roboflow source image appearing in different splits
    """
    df = df.reset_index(drop=True)
    md5s = [_md5(p) for p in df.path]
    phashes = []
    for p in df.path:
        with Image.open(p) as im:
            phashes.append(imagehash.phash(im.convert("RGB"), hash_size=config.PHASH_SIZE))

    rows = []
    for i, j in itertools.combinations(range(len(df)), 2):
        a, b = df.iloc[i], df.iloc[j]
        dist = int(phashes[i] - phashes[j])
        exact = md5s[i] == md5s[j]
        cross = a.split != b.split
        same_source = cross and a.source_id == b.source_id
        if exact or dist <= threshold or same_source:
            rows.append(
                dict(
                    path_a=a.path, split_a=a.split, label_a=a.label,
                    path_b=b.path, split_b=b.split, label_b=b.label,
                    hamming=dist, exact=exact, same_source=same_source,
                    cross_split=cross, label_conflict=a.label != b.label,
                )
            )
    return pd.DataFrame(rows)


def remove_leakage(df: pd.DataFrame, pairs: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Drop leaked images so valid/test stay pristine.

    * Cross-split pair -> the copy in the lower-priority split is removed
      (priority: test > valid > train).
    * Byte-identical copies inside one split -> keep the first, drop the rest.
    """
    if pairs.empty:
        return df.copy(), []
    rank = {"train": 0, "valid": 1, "test": 2}
    drop = set()
    for r in pairs.itertuples():
        if r.cross_split:
            drop.add(r.path_a if rank[r.split_a] < rank[r.split_b] else r.path_b)
        elif r.exact:
            drop.add(r.path_b)
    kept = df[~df.path.isin(drop)].reset_index(drop=True)
    return kept, sorted(drop)


def leakage_report(df: pd.DataFrame, remove: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the duplicate scan, print a summary, optionally clean the manifest."""
    pairs = find_duplicate_pairs(df)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(config.METRICS_DIR / "duplicate_pairs.csv", index=False)

    if pairs.empty:
        print("\n[leakage] No exact/near-duplicate images found. Splits look clean.")
        return df, pairs

    cross = pairs[pairs.cross_split]
    print("\n[leakage] Duplicate scan summary")
    print(f"  total flagged pairs     : {len(pairs)}")
    print(f"  cross-split (LEAKAGE)   : {len(cross)}")
    print(f"  exact byte duplicates   : {int(pairs.exact.sum())}")
    print(f"  same-source cross-split : {int(pairs.same_source.sum())}")
    print(f"  conflicting labels      : {int(pairs.label_conflict.sum())}")
    if pairs.label_conflict.any():
        print("  WARNING: near-identical images carry different labels -> inspect duplicate_pairs.csv")

    if not remove:
        print("  remove_leaked=False -> manifest left untouched.")
        return df, pairs

    cleaned, dropped = remove_leakage(df, pairs)
    print(f"  removed {len(dropped)} image(s) from train/valid; {len(cleaned)} images remain")
    return cleaned, pairs


# --------------------------------------------------------------------------- #
# 5. tf.data pipeline
# --------------------------------------------------------------------------- #
def build_augmenter() -> keras.Sequential:
    """Mild, medically-plausible augmentation applied to training batches only."""
    layers = []
    if config.AUG_HORIZONTAL_FLIP:
        layers.append(keras.layers.RandomFlip("horizontal"))
    layers += [
        keras.layers.RandomRotation(config.AUG_ROTATION, fill_mode="reflect"),
        keras.layers.RandomZoom(config.AUG_ZOOM, fill_mode="reflect"),
        keras.layers.RandomTranslation(config.AUG_TRANSLATION, config.AUG_TRANSLATION, fill_mode="reflect"),
        keras.layers.RandomContrast(config.AUG_CONTRAST),
        keras.layers.RandomBrightness(config.AUG_BRIGHTNESS, value_range=(0, 255)),
    ]
    return keras.Sequential(layers, name="augmentation")


def _decode(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, config.IMG_SIZE, antialias=True)
    return tf.cast(img, tf.float32), tf.cast(label, tf.float32)


def make_dataset(
    df: pd.DataFrame,
    training: bool,
    batch_size: int = config.BATCH_SIZE,
    augment: bool = True,
) -> tf.data.Dataset:
    """Build a dataset yielding ``(image[0-255 float32], label)``.

    Pixel scaling / backbone-specific preprocessing happens *inside* each model so
    saved ``.keras`` files accept raw images. Evaluation datasets keep the
    manifest order so predictions can be mapped back to file paths.
    """
    paths = df.path.astype(str).tolist()
    labels = df.label.astype("float32").tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augment:
        aug = build_augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def build_datasets(splits: Dict[str, pd.DataFrame]) -> Dict[str, tf.data.Dataset]:
    return {
        "train": make_dataset(splits["train"], training=True),
        "valid": make_dataset(splits["valid"], training=False),
        "test": make_dataset(splits["test"], training=False),
    }


# --------------------------------------------------------------------------- #
# 6. One-call preparation
# --------------------------------------------------------------------------- #
def prepare_data(
    api_key: Optional[str] = None,
    remove_leaked: bool = True,
    do_eda: bool = True,
    show: bool = True,
) -> dict:
    """Download -> manifest -> EDA -> leakage check -> per-split DataFrames."""
    config.ensure_dirs()
    root = download_dataset(api_key=api_key)
    manifest = build_manifest(root)
    if do_eda:
        manifest = run_eda(manifest, show=show)
    manifest, pairs = leakage_report(manifest, remove=remove_leaked)
    splits = {s: manifest[manifest.split == s].reset_index(drop=True) for s in config.SPLITS}
    for s, d in splits.items():
        if d.label.nunique() < 2:
            raise RuntimeError(f"Split '{s}' contains a single class after cleaning - check the dataset.")
    manifest.to_csv(config.METRICS_DIR / "manifest_clean.csv", index=False)
    return {"root": root, "manifest": manifest, "splits": splits, "duplicates": pairs}
