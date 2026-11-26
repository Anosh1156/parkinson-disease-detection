"""Training loops, callbacks, class weights and two-phase fine-tuning.

Run end-to-end from the repo root::

    python -m src.train --models custom_cnn vgg16 resnet50
"""
from __future__ import annotations

import argparse
import gc
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from . import config, data_loader, models


def configure_gpu() -> None:
    """Enable memory growth and report available devices."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"[gpu] TensorFlow {tf.__version__} | GPUs: {[g.name for g in gpus] or 'none (CPU only)'}")


def compute_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """Balanced class weights computed from the *training* split only."""
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=train_df.label.values)
    out = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[train] class weights: {out}")
    return out


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def _callbacks(name: str, tag: str, patience: int):
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = str(config.CHECKPOINT_DIR / f"{name}_{tag}.weights.h5")
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE, min_lr=config.MIN_LR, verbose=1,
        ),
        # Best weights are saved explicitly and reloaded after fit(): robust across Keras versions.
        keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
    ]
    return cbs, ckpt


def _fit_phase(model, train_ds, val_ds, epochs, lr, class_weight, name, tag, patience) -> Tuple[pd.DataFrame, float]:
    compile_model(model, lr)
    cbs, ckpt = _callbacks(name, tag, patience)
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=class_weight, callbacks=cbs, verbose=2,
    )
    model.load_weights(ckpt)  # guarantee we continue from / return the best epoch
    df = pd.DataFrame(hist.history)
    df["epoch"] = np.arange(1, len(df) + 1)
    df["phase"] = tag
    return df, float(df["val_loss"].min())


def train_model(
    name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: Dict[int, float],
) -> Tuple[keras.Model, pd.DataFrame]:
    """Train one architecture.

    * ``custom_cnn``: single phase from scratch.
    * ``vgg16`` / ``resnet50``: phase 1 trains the new head with the backbone frozen;
      phase 2 unfreezes the top backbone layers at a much lower learning rate. The
      better phase (by validation loss) is kept.
    """
    config.set_global_seed()
    model = models.get_model(name)
    print(f"\n{'=' * 70}\nTraining {name}  ({model.count_params():,} params)\n{'=' * 70}")

    if not models.is_transfer_model(name):
        hist, _ = _fit_phase(model, train_ds, val_ds, config.EPOCHS_SCRATCH, config.LR_SCRATCH,
                             class_weight, name, "scratch", config.EARLY_STOP_PATIENCE_SCRATCH)
        history = hist
    else:
        print("[train] Phase 1: frozen backbone, training classification head")
        h1, loss1 = _fit_phase(model, train_ds, val_ds, config.EPOCHS_HEAD, config.LR_HEAD,
                               class_weight, name, "head", config.EARLY_STOP_PATIENCE)

        n = models.unfreeze_top_layers(model, name)
        print(f"[train] Phase 2: fine-tuning top {n} backbone layers (lr={config.LR_FINE_TUNE})")
        h2, loss2 = _fit_phase(model, train_ds, val_ds, config.EPOCHS_FINE_TUNE, config.LR_FINE_TUNE,
                               class_weight, name, "fine_tune", config.EARLY_STOP_PATIENCE)
        if loss2 > loss1:  # fine-tuning did not help -> fall back to the phase-1 weights
            print(f"[train] Fine-tune val_loss {loss2:.4f} > head val_loss {loss1:.4f}; reverting to phase 1")
            model.load_weights(str(config.CHECKPOINT_DIR / f"{name}_head.weights.h5"))
        history = pd.concat([h1, h2], ignore_index=True)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODELS_DIR / f"{name}.keras")
    history.to_csv(config.METRICS_DIR / f"history_{name}.csv", index=False)
    return model, history


def train_all(
    model_names: Iterable[str],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: Dict[int, float],
) -> Tuple[Dict[str, keras.Model], Dict[str, pd.DataFrame]]:
    trained, histories = {}, {}
    for name in model_names:
        trained[name], histories[name] = train_model(name, train_ds, val_ds, class_weight)
        gc.collect()
    return trained, histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Parkinson's detection models")
    parser.add_argument("--models", nargs="+", default=list(config.MODEL_NAMES), choices=list(models.MODEL_REGISTRY))
    parser.add_argument("--keep-leaked", action="store_true", help="do not remove cross-split duplicates")
    parser.add_argument("--no-eda", action="store_true")
    args = parser.parse_args()

    configure_gpu()
    config.set_global_seed()
    data = data_loader.prepare_data(remove_leaked=not args.keep_leaked, do_eda=not args.no_eda, show=False)
    ds = data_loader.build_datasets(data["splits"])
    cw = compute_class_weights(data["splits"]["train"])
    train_all(args.models, ds["train"], ds["valid"], cw)
    print("\nDone. Run `python -m src.evaluate` to score the saved models on the test set.")


if __name__ == "__main__":
    main()
