# Fire and Smoke Detection & Dataset Pipeline

A modular Python framework for auditing fire & smoke datasets, preparing leakage-free train/validation/test splits, and training YOLO-based object detection models for fire and smoke detection.

---

## Overview

The project is organized into two primary packages:

1. **`fire_audit` (`src/fire_audit`)**:
   - **Audit**: Scans images and bounding box annotations for corruption, coordinate errors, and class balance.
   - **Segment**: Identifies continuous video sequences and near-duplicate frames using perceptual hashing (`dHash`).
   - **Prepare**: Partitions dataset sequences into train, validation, and test splits to prevent data leakage.
   - **Verify**: Acceptance testing on split integrity and YOLO annotation formats.

2. **`fire_detection` (`src/fire_detection`)**:
   - **Train**: Fine-tunes pretrained YOLO models on the prepared fire & smoke dataset.
   - **Evaluate**: Evaluates model performance (Precision, Recall, mAP50, mAP50-95).
   - **Infer**: Runs visual object detection inference on test images or directories.

3. **`modules` (`src/modules`)**:
   - Custom modules for training

---

## Directory Structure

```text
├── data.yaml            # YOLO dataset configuration (classes, split paths)
├── evaluate.py          # CLI to evaluate model checkpoints
├── infer.py             # CLI to run detection inference on images
├── run_pipeline.py      # CLI for dataset audit, segmentation, and preparation
├── train.py             # CLI to train/fine-tune YOLO models
├── datasets/            # Multiple datasets (including manifest and split folders), added to .gitignore
├── runs/
│   ├── infer/           # Inference results
│   ├── train/           # Training results
│   └── val/             # Validation results
│
└── src/
    ├── fire_audit/      # Dataset audit, sequence segmentation & partitioning
    ├── fire_detection/  # Model training, evaluation & inference routines
    └── modules/         # Custom bbox loss, detection loss, detection model & trainer
```

---

## Getting Started

### Dataset

We uses multiple datasets for testing.

Download our training dataset here: https://www.kaggle.com/datasets/dtashton/combined-dataset

### Shared Runtime Configuration

Machine-specific dataset and training values are kept in the project-root `config.yaml` file. 
Edit that file once to set `data_path`, `training.weights`, and the training parameters. 
The loader also accepts an alternate location through the `CONFIG` environment variable.

### 1. Dataset Audit and Preparation

Run the full dataset pipeline (audit, segmentation, partitioning, and verification):

```bash
python run_pipeline.py all --data-dir "path/to/dataset" --output-dir "path/to/output" --train-ratio 0.8 --seed 42
```

Or run individual stages:

```bash
# Audit images and annotations
python run_pipeline.py audit --data-dir "path/to/dataset" --output-dir "path/to/output"

# Segment video frames to prevent data leakage
python run_pipeline.py segment --data-dir "path/to/dataset" --output-dir "path/to/output"

# Partition into train/val/test splits
python run_pipeline.py prepare --data-dir "path/to/dataset" --train-ratio 0.8
```

---

### 2. Model Training

Fine-tune a YOLO model on the dataset specified in `data.yaml`:

```bash
python train.py --weights "path/to/pretrained_model.pt" --data data.yaml --epochs 50 --batch 16 --imgsz 640
```

#### YOLO26n with P2, BiFPN, and spatial-relational co-attention

`yolo26n-p2_bifpn_src.yaml` is the project's configured architecture for fire and smoke detection. Its YOLO26n backbone layers 0–10 are unchanged, allowing compatible pretrained weights to transfer. The 64-channel BiFPN fuses P5→P2 and P2→P5 with learned, normalized nonnegative weights. Spatial-relational co-attention is placed only in the neck, after the fused and refined P2 top-down and P4 bottom-up nodes. The stock YOLO26 `Detect` head and end-to-end loss receive four scales with strides 4, 8, 16, and 32.

```bash
python train.py --model-yaml yolo26n-p2_bifpn_src.yaml --weights yolo26n-p2.pt --loss ciou --name yolo26n-p2_bifpn_src --imgsz 640
```

The CLI registers the custom modules before Ultralytics reads the YAML. For direct Python use, import `register_modules` from `src.modules.bifpn_src`, call it, then create `YOLO("yolo26n-p2_bifpn_src.yaml")`. `config.yaml` selects this model by default.

The YAML targets the `n` scale and an input size divisible by 32. To change attention locations, add or remove `SpatialRelationalCoAttentionLite` nodes in the neck and update downstream `from` indices and channel arguments. The fusion width is 64 actual channels at scale `n`; changing it also requires updating the `Index`, fusion, and refinement channel arguments. `WeightedBiFPNFusion` takes a concatenated set of equal-width features and returns their normalized weighted sum; the following `Index` tells Ultralytics' parser that the sum has 64 channels. The attention module caps relational attention at an 8×8 token grid while its spatial gate remains at feature resolution. Its signed spatial/channel modulation can enhance or reduce local feature responses, although a reduction in fire/smoke false alarms has not been measured. Relational attention operates on the already fused map; it is not explicit pairwise cross-attention between source maps.

With Ultralytics 8.4.140, two classes, and a 640×640 input, the neck-only SRC model has 1,609,078 parameters and about 6.989 THOP GFLOPs. Stock YOLO26n-P2 has 2,516,240 parameters and about 7.62 THOP GFLOPs; three-scale YOLO26n has 2,504,580 parameters and about 5.89 THOP GFLOPs. The P2 branch adds compute relative to the three-scale model. THOP may omit functional attention and fusion work, so benchmark target hardware before making a speed claim.

Run `python src/modules/bifpn_src/validate.py` to check the four scales, feature shapes, stock end-to-end loss, and gradients without a dataset or checkpoint.

---

### 3. Model Evaluation

Evaluate trained weights on validation or test sets:

```bash
python evaluate.py --weights "runs/train/model_name/weights/best.pt" --data data.yaml --split test --out-dir runs/val/eval
```

Use `--split test` once `data.yaml` points to an available test split. 

The run writes `metrics.json` and `metrics.csv` with the usual detection metrics plus `size_ap.small` and `size_ap.tiny` (CSV categories `small/all_classes`, `small/fire`, `small/smoke`, and the corresponding `tiny` rows). Small uses ground-truth boxes with square-root area >= 32 pixels; Tiny uses >= 16 pixels.

---

### 4. Inference and Visualization

Run detection on sample images:

```bash
python infer.py --weights "runs/train/model_name/weights/best.pt" --source "path/to/images" --output-dir "runs/infer"
```
