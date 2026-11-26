# Parkinson's Disease Detection (Image Classification)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/parkinson-disease-detection/blob/main/notebooks/Parkinson_Detection_Colab.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)

Binary image classification (`healthy` vs `parkinson`) on a very small Roboflow dataset
(~189 images). The project compares three architectures — a compact **custom CNN**,
**VGG16** and **ResNet50** (ImageNet transfer learning) — with a pipeline designed around
the two main risks of tiny datasets: **overfitting** and **data leakage**.

> **Medical disclaimer.** This is a research / educational project. It is **not** a medical
> device, has not been clinically validated, and must **not** be used to diagnose, screen
> for, or make decisions about Parkinson's disease. Consult a qualified clinician for any
> health concern.

## Dataset

| | |
|---|---|
| Source | Roboflow Universe — workspace `anosh-goodwin`, project `parkinson-disease-detection-1eeyh` |
| Task | Binary classification: `healthy` (0) vs `parkinson` (1) |
| Size | ~189 images |
| Export | `folder` format → `train/`, `valid/`, `test/` with one sub-folder per class |

Set `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT` and `ROBOFLOW_VERSION` in `src/config.py` to the dataset version you want. You need your own
Roboflow API key; it is **never** stored in the repo.

## Pipeline

```
Roboflow API ──► manifest (path, split, label, source id)
                      │
                      ├─► EDA: class balance, image sizes, sample grid
                      ├─► Leakage check: MD5 + perceptual hash + Roboflow source id
                      │      └─ cross-split duplicates removed from train/valid (test untouched)
                      ▼
            tf.data: decode → resize 224×224 → cache → shuffle → batch → augment (train only)
                      ▼
   ┌──────────────┬──────────────────────┬──────────────────────┐
   │ Custom CNN   │ VGG16 (frozen→top 4) │ ResNet50 (frozen→top 30) │
   └──────┬───────┴──────────┬───────────┴───────────┬──────────┘
          │  class weights · EarlyStopping · ReduceLROnPlateau · best-weights checkpoint
          ▼
   Evaluation (test set used once):
   accuracy · balanced acc · precision · recall · specificity · F1 · MCC · ROC/PR AUC
   bootstrap 95% CIs · confusion matrices · ROC curves · error analysis
   model choice and decision threshold (Youden's J) picked on the validation set only
```

### Design choices for a tiny dataset

- **Leakage control** – perceptual-hash duplicate scan across all splits (plus Roboflow
  `.rf.` source-id matching); leaked copies are removed from train/valid so the test set stays
  clean. The report is saved to `outputs/metrics/duplicate_pairs.csv`.
- **No test-set tuning** – early stopping, LR schedule, threshold tuning and best-model
  selection all use the validation split.
- **Class weights** computed from the training split only.
- **Small custom CNN** (4 conv blocks, BatchNorm, SpatialDropout, GAP) instead of a deep net.
- **Two-phase transfer learning** – train the head with a frozen backbone, then unfreeze the
  top layers at a 100× lower learning rate (BatchNorm layers stay frozen). If fine-tuning does
  not improve validation loss, phase-1 weights are restored.
- **Preprocessing inside the model** – saved `.keras` files accept raw 0–255 RGB images.
- **Bootstrap confidence intervals** – with a test set of a few dozen images, a single accuracy
  number is not trustworthy; the CIs show how wide the uncertainty is.
- **Mild augmentation** (flip, rotation, zoom, shift, contrast, brightness) on training data
  only. If your images are orientation-sensitive (e.g. spiral direction), set
  `AUG_HORIZONTAL_FLIP = False` in `src/config.py`.

## Quick start (Google Colab)

1. Click the **Open in Colab** badge (after replacing `YOUR_GITHUB_USERNAME`), or upload
   `notebooks/Parkinson_Detection_Colab.ipynb` to Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Add your key under **Secrets** (key icon, left sidebar) as `ROBOFLOW_API_KEY` and enable
   notebook access. If no secret is found, the notebook prompts for it securely.
4. `Runtime → Run all`.

The notebook is self-contained: it writes every `src/*.py` file itself, so it also works
without cloning the repo. Outputs (models, plots, metrics) are zipped for download at the end.

## Local setup

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/parkinson-disease-detection.git
cd parkinson-disease-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export ROBOFLOW_API_KEY="your_key_here"      # Windows: set ROBOFLOW_API_KEY=...
python -m src.train                          # trains custom_cnn, vgg16, resnet50
python -m src.evaluate                       # scores saved models on the test set
# options: --models vgg16 resnet50   --keep-leaked   --no-eda
```

## Repository layout

```
parkinson-disease-detection/
├── notebooks/Parkinson_Detection_Colab.ipynb   # end-to-end Colab notebook
├── src/
│   ├── config.py        # hyperparameters, paths, seeds, API-key lookup
│   ├── data_loader.py   # download, EDA, leakage check, tf.data + augmentation
│   ├── models.py        # custom CNN, VGG16, ResNet50
│   ├── train.py         # callbacks, class weights, two-phase fine-tuning
│   └── evaluate.py      # metrics, comparison table, plots, error analysis
├── outputs/             # models (.keras), plots, metrics CSVs (generated)
├── requirements.txt
└── .gitignore
```

## Outputs

| Path | Content |
|---|---|
| `outputs/models/*.keras` | trained models (git-ignored) |
| `outputs/metrics/model_comparison.csv` | full metrics table, both thresholds, bootstrap CIs |
| `outputs/metrics/duplicate_pairs.csv` | flagged duplicate / leakage pairs |
| `outputs/metrics/predictions_<model>.csv` | per-image test predictions |
| `outputs/plots/*.png` | EDA, training curves, ROC, confusion matrices, error analysis |

## Results

Run the pipeline and paste your `model_comparison.csv` summary here. Report the confidence
intervals alongside point estimates — on ~189 images, differences between models are often
within noise.

## Limitations

- Tiny dataset; results can vary substantially with the random seed and split.
- Image provenance (device, patient, session) is unknown, so patient-level leakage cannot be
  fully ruled out; the duplicate scan only catches visually identical or near-identical images.
- No external validation cohort; no clinical validation of any kind.

## License

Add a license of your choice (e.g. MIT) and respect the dataset's own license on Roboflow Universe.
