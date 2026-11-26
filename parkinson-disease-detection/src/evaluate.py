"""Metrics, comparison table, plots, confusion matrices, ROC curves and error analysis.

Model selection and threshold tuning use the *validation* split only; the test split is
touched exactly once for final reporting.

Run from the repo root to score saved models::

    python -m src.evaluate
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from tensorflow import keras

from . import config, data_loader, models  # noqa: F401  (models import registers custom layers)


# --------------------------------------------------------------------------- #
# Predictions & metrics
# --------------------------------------------------------------------------- #
def predict_probs(model: keras.Model, ds) -> np.ndarray:
    """Probabilities in dataset order (evaluation datasets are never shuffled)."""
    return model.predict(ds, verbose=0).ravel()


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (chosen on validation data)."""
    if len(np.unique(y_true)) < 2:
        return config.DEFAULT_THRESHOLD
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[best], 0.0, 1.0))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),            # sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if two_classes else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if two_classes else np.nan,
        "brier": brier_score_loss(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, y_prob, fn, n: int = config.N_BOOTSTRAP, seed: int = config.SEED):
    """95% percentile bootstrap CI - essential with a test set of only a few dozen images."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(fn(y_true[idx], y_prob[idx]))
    if not vals:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_models(trained: Dict[str, keras.Model], splits: Dict[str, pd.DataFrame], datasets: dict):
    """Score every model on valid + test. Returns (comparison_table, predictions)."""
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    y_val, y_test = splits["valid"].label.values, splits["test"].label.values
    rows, preds = [], {}
    for name, model in trained.items():
        p_val, p_test = predict_probs(model, datasets["valid"]), predict_probs(model, datasets["test"])
        thr_tuned = youden_threshold(y_val, p_val)
        preds[name] = {"val": p_val, "test": p_test, "thr_tuned": thr_tuned}
        val_m = compute_metrics(y_val, p_val, config.DEFAULT_THRESHOLD)
        for mode, thr in (("0.5", config.DEFAULT_THRESHOLD), ("val_tuned", thr_tuned)):
            m = compute_metrics(y_test, p_test, thr)
            acc_ci = bootstrap_ci(y_test, p_test, lambda y, p, t=thr: accuracy_score(y, (p >= t).astype(int)))
            auc_ci = bootstrap_ci(y_test, p_test, roc_auc_score)
            rows.append({
                "model": name, "threshold_mode": mode, "threshold": round(thr, 4),
                **m,
                "accuracy_ci95": f"[{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]",
                "auc_ci95": f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]",
                "val_auc": val_m["roc_auc"], "val_accuracy": val_m["accuracy"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    return table, preds


def select_best_model(table: pd.DataFrame) -> str:
    """Pick the winner from *validation* AUC (ties -> validation accuracy), never from test."""
    t = table.drop_duplicates("model").sort_values(["val_auc", "val_accuracy"], ascending=False)
    return str(t.iloc[0]["model"])


def summary_view(table: pd.DataFrame, mode: str = "0.5") -> pd.DataFrame:
    cols = ["model", "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
            "f1", "roc_auc", "pr_auc", "mcc", "accuracy_ci95", "auc_ci95", "val_auc"]
    return (table[table.threshold_mode == mode][cols]
            .sort_values("val_auc", ascending=False).round(3).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _finish(fig, filename: str, show: bool = True) -> None:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_training_curves(histories: Dict[str, pd.DataFrame], show: bool = True) -> None:
    fig, axes = plt.subplots(len(histories), 3, figsize=(16, 3.8 * len(histories)), squeeze=False)
    for r, (name, h) in enumerate(histories.items()):
        x = np.arange(1, len(h) + 1)
        for c, (metric, title) in enumerate((("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC"))):
            ax = axes[r, c]
            ax.plot(x, h[metric], label="train")
            ax.plot(x, h[f"val_{metric}"], label="valid")
            changes = np.where(h["phase"].values[1:] != h["phase"].values[:-1])[0]
            for ch in changes:
                ax.axvline(ch + 1.5, color="grey", ls="--", lw=1)
                ax.text(ch + 1.6, ax.get_ylim()[0], " fine-tune", fontsize=8, va="bottom")
            ax.set_title(f"{name} - {title}")
            ax.set_xlabel("epoch")
            ax.legend()
    _finish(fig, "training_curves.png", show)


def plot_confusion_matrices(preds: dict, splits: dict, show: bool = True) -> None:
    y = splits["test"].label.values
    names = list(preds)
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7.5), squeeze=False)
    for c, name in enumerate(names):
        for r, (mode, thr) in enumerate((("0.5", config.DEFAULT_THRESHOLD), ("val-tuned", preds[name]["thr_tuned"]))):
            cm = confusion_matrix(y, (preds[name]["test"] >= thr).astype(int), labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[r, c],
                        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
            axes[r, c].set_title(f"{name}\nthreshold={thr:.2f} ({mode})")
            axes[r, c].set_xlabel("predicted")
            axes[r, c].set_ylabel("true")
    _finish(fig, "confusion_matrices.png", show)


def plot_roc_curves(preds: dict, splits: dict, show: bool = True) -> None:
    y = splits["test"].label.values
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p["test"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y, p['test']):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)", title="ROC curves - test set")
    ax.legend(loc="lower right")
    _finish(fig, "roc_curves.png", show)


def plot_metric_comparison(table: pd.DataFrame, mode: str = "0.5", show: bool = True) -> None:
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    long = table[table.threshold_mode == mode].melt(id_vars="model", value_vars=metrics,
                                                    var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=long, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Test-set metrics (threshold mode: {mode})")
    _finish(fig, "metric_comparison.png", show)


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def error_analysis(name: str, test_df: pd.DataFrame, probs: np.ndarray,
                   threshold: float = config.DEFAULT_THRESHOLD, max_images: int = 12,
                   show: bool = True) -> pd.DataFrame:
    """Save per-image predictions and plot misclassified images (FN are the costly errors)."""
    out = test_df[["path", "class_name", "label"]].copy()
    out["p_parkinson"] = probs
    out["pred"] = (probs >= threshold).astype(int)
    out["correct"] = out.pred == out.label
    out["error_type"] = np.where(out.correct, "ok", np.where(out.label == 1, "false_negative", "false_positive"))
    out["confidence"] = np.where(out.pred == 1, probs, 1 - probs)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.METRICS_DIR / f"predictions_{name}.csv", index=False)

    errs = out[~out.correct].sort_values("confidence", ascending=False)
    print(f"[errors] {name}: {len(errs)} / {len(out)} misclassified "
          f"(FN={int((errs.error_type == 'false_negative').sum())}, FP={int((errs.error_type == 'false_positive').sum())})")
    if errs.empty:
        print("[errors] No misclassifications on the test set.")
        return out

    errs = errs.head(max_images)
    cols = min(4, len(errs))
    rows = int(np.ceil(len(errs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), errs.iterrows()):
        ax.imshow(Image.open(r.path).convert("RGB").resize(config.IMG_SIZE))
        ax.set_title(f"true: {r.class_name}\n{r.error_type} (p={r.p_parkinson:.2f})", fontsize=9, color="crimson")
    fig.suptitle(f"{name}: most confident misclassifications", y=1.02)
    _finish(fig, f"error_analysis_{name}.png", show)
    return out


# --------------------------------------------------------------------------- #
# CLI: evaluate saved models
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved models on the test split")
    parser.add_argument("--models", nargs="+", default=list(config.MODEL_NAMES))
    args = parser.parse_args()

    config.set_global_seed()
    data = data_loader.prepare_data(do_eda=False, show=False)
    ds = data_loader.build_datasets(data["splits"])
    trained = {n: keras.models.load_model(config.MODELS_DIR / f"{n}.keras") for n in args.models}

    table, preds = evaluate_models(trained, data["splits"], ds)
    print(summary_view(table, "0.5").to_string(index=False))
    print("\nVal-tuned threshold:\n", summary_view(table, "val_tuned").to_string(index=False))

    plot_roc_curves(preds, data["splits"], show=False)
    plot_confusion_matrices(preds, data["splits"], show=False)
    plot_metric_comparison(table, show=False)
    best = select_best_model(table)
    print(f"\nBest model by validation AUC: {best}")
    error_analysis(best, data["splits"]["test"], preds[best]["test"], show=False)


if __name__ == "__main__":
    main()
