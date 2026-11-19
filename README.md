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

#### Three YOLO26n-P2 architecture experiments

All three models use the unchanged YOLO26n-P2 backbone, four P2-P5 detection scales with strides 4/8/16/32, the stock end-to-end `Detect` head, and the standard detection loss. The experiment changes are confined to the neck:

| Experiment | YAML | Neck | SRC positions | Detect inputs |
| --- | --- | --- | --- | --- |
| 1. BiFPN | `yolo26n-p2_bifpn.yaml` | One six-node weighted bidirectional P2-P5 BiFPN cell | None | `[29, 34, 39, 44]` |
| 2. SRC | `yolo26n-p2_src.yaml` | Stock YOLO26n-P2 PAN | Fused P2/P4, layers `[20, 27]` | `[20, 23, 27, 30]` |
| 3. BiFPN + SRC | `yolo26n-p2_bifpn_src.yaml` | The same six-node BiFPN topology as experiment 1 | Fused P2/P4, layers `[30, 41]` | `[30, 35, 41, 46]` |

Experiments 2 and 3 use `NeckSpatialRelationCoAttention` in `src/modules/co_attention/neck.py`, adapted from the sample `src/modules/co_attention/src_coattention.py`. The sample's path uses an underscore (`co_attention`). The module creates latent fire and smoke branches and exchanges information in both directions: full-resolution queries read keys/values pooled to at most 8x8, with query chunks limiting the attention working set. It preserves feature shape. These latent gates are learned through the standard detection loss; the sample's box-occupancy auxiliary loss and custom trainer are not used. The branch names therefore do not imply independently supervised fire/smoke maps.

Use the same data, pretrained weights, seed, image size, epochs, batch size, and loss when comparing the three runs. Give every run a distinct name because `config.yaml` sets `exist_ok: true`:

```bash
python train.py --model-yaml yolo26n-p2_bifpn.yaml --weights yolo26n-p2.pt --data data.yaml --loss ciou --seed 42 --imgsz 640 --epochs 200 --batch 16 --name exp1_bifpn
python train.py --model-yaml yolo26n-p2_src.yaml --weights yolo26n-p2.pt --data data.yaml --loss ciou --seed 42 --imgsz 640 --epochs 200 --batch 16 --name exp2_src
python train.py --model-yaml yolo26n-p2_bifpn_src.yaml --weights yolo26n-p2.pt --data data.yaml --loss ciou --seed 42 --imgsz 640 --epochs 200 --batch 16 --name exp3_bifpn_src_coattention
```

`config.yaml` continues to select experiment 3 by default. The train, infer, and evaluate entry points register both custom module packages before model loading. With Ultralytics 8.4.140, two classes, and a 640x640 input, the architecture measurements are:

| Experiment | Parameters | THOP GFLOPs | Matching checkpoint state items |
| --- | ---: | ---: | ---: |
| 1. BiFPN | 1,604,766 | 6.9833 | 240/632 |
| 2. SRC | 2,553,638 | 8.1261 | 402/950 |
| 3. BiFPN + SRC | 1,638,068 | 7.6702 | 240/680 |

All three transfer the same 240 backbone state items. Experiment 2 also transfers matching stock PAN/head items, so initialization beyond the backbone differs between experiments. THOP may omit functional scaled dot-product attention and fusion work; the numbers are not deployment latency or memory measurements. No comparative training metrics have been collected yet.

Run `python src/fire_detection/validate_experiments.py` to check the three YAML graphs, backbone identity, BiFPN topology, four detection scales, stock loss, and gradients without a dataset or checkpoint.

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
