# VisionGuard — Manufacturing Defect Detection System

**Code A Nova | Machine Learning Internship Project**

VisionGuard is an educational binary image-classification project that classifies product images as `normal` or `defective`. It uses PyTorch transfer learning with ResNet-18, image augmentation, TensorBoard, checkpointing, evaluation metrics, and single-image inference.

> This is an educational demonstration, not a certified industrial inspection system. Results depend on the dataset and must be independently validated before any real production use.

## 1. Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- VS Code
- Internet connection for first-time package installation and pretrained weights
- CPU is supported; an NVIDIA GPU can accelerate training

## 2. Extract and open

1. Extract `VisionGuard_Ready_to_Run.zip`.
2. In VS Code, choose **File → Open Folder** and select `VisionGuard_Ready_to_Run`.
3. Open **Terminal → New Terminal** (PowerShell).

## 3. Create an environment and install packages

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is unavailable, use `python -m venv .venv` with an installed supported Python version.

## 4. Prepare the dataset

### Option A — Your own labeled dataset

Create this structure and put image files in the matching class folders:

```text
data/
  raw/
    normal/
      image001.jpg
      ...
    defective/
      image101.jpg
      ...
```

The class folder names must be exactly `normal` and `defective` (case-insensitive). Supported formats: JPG, JPEG, PNG, BMP, TIF, TIFF.

### Option B — MVTec AD

1. Download MVTec AD from the official dataset page: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Extract it.
3. Run the preparation script for one category (for example `bottle`):

```powershell
python scripts/prepare_mvtec.py --dataset-root "C:\path\to\mvtec_anomaly_detection" --category bottle
```

The script combines that category's `train/good`, `test/good`, and defect images into `data/raw/normal` and `data/raw/defective`. It copies images; it does not copy pixel-level masks. The official brief names MVTec AD or custom synthetic defect images as dataset options.

### Split the data

```powershell
python scripts/split_dataset.py --source data/raw --output data/split --train 0.70 --val 0.15 --test 0.15
```

The script checks that both classes have enough images and creates a stratified split. It uses a fixed seed for reproducibility. Do not put the same or near-duplicate image in multiple source files.

Expected result:

```text
data/split/
  train/normal/       train/defective/
  val/normal/         val/defective/
  test/normal/        test/defective/
```

## 5. Train ResNet-18

```powershell
python train.py --data-dir data/split --epochs 15 --batch-size 16 --image-size 224
```

Useful options:

```powershell
python train.py --help
```

- `--epochs`: maximum number of epochs
- `--batch-size`: lower this (for example 8) if memory is limited
- `--lr`: learning rate
- `--patience`: early-stopping patience
- `--no-pretrained`: do not download/use ImageNet pretrained weights
- `--resume`: resume from `outputs/checkpoints/last.pth`

Training saves:
- `outputs/checkpoints/best.pth` — best validation checkpoint
- `outputs/checkpoints/last.pth` — latest checkpoint, including optimizer state
- `outputs/training_history.csv`
- TensorBoard logs in `runs/visionguard`

## 6. View TensorBoard

In a second VS Code terminal (with the environment activated):

```powershell
tensorboard --logdir runs
```

Open the local URL printed in the terminal (usually http://localhost:6006).

## 7. Evaluate the held-out test set

```powershell
python evaluate.py --data-dir data/split --checkpoint outputs/checkpoints/best.pth
```

Reports are saved under `outputs/reports/`:
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png`

Metrics include accuracy, precision, recall, F1-score, ROC-AUC (when both test classes are present), and per-class results. `defective` is treated as the positive class.

## 8. Predict one image

```powershell
python predict.py --image "C:\path\to\product.jpg" --checkpoint outputs/checkpoints/best.pth
```

The script prints the predicted class and class probabilities. It can run without the original dataset folder.

## 9. Project structure

```text
VisionGuard_Ready_to_Run/
  train.py
  evaluate.py
  predict.py
  requirements.txt
  src/
    data.py
    models.py
    utils.py
  scripts/
    split_dataset.py
    prepare_mvtec.py
  docs/
    7_DAY_PLAN.md
    DATASET_GUIDE.md
  tests/
    test_project.py
  outputs/                 (created when you run the project)
  runs/                    (created by TensorBoard during training)
```

## 10. Important notes

- The requested 70/15/15 split is applied to the images supplied in `data/raw`.
- The MVTec preparation script supports one category at a time. Run it separately for each category only if you intentionally want a combined dataset; avoid mixing categories without documenting the experiment.
- A target of >90% accuracy, >85% defective-class recall, and >0.92 AUC is stated in the internship brief. These are evaluation goals, not guaranteed outcomes. Report actual held-out test results honestly.
- If the dataset is very small, imbalanced, or visually diverse, more data and tuning may be needed.
- The MVTec AD dataset includes pixel masks, but this starter project performs image-level classification and does not train a segmentation model.
