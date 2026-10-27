# YOLO-SR: A Zero-Cost Super-Resolution and Edge-Aware Architecture for Micro-Scale UAV Object Detection

Official implementation of the paper **"A Zero-Cost Super-Resolution and Edge-Aware Architecture for Micro-Scale UAV Object Detection."**

YOLO-SR is an edge-aware dual-path detector built on the **YOLO26n** baseline. Every component that recovers fine spatial detail: a Super-Resolution Auxiliary Branch (SAB), a Reparameterised Contextual Fusion (RepCF) module, and a Structure-Aware Multi-Task Loss  is active **only during training** and is folded away before deployment, so the deployed model is computationally identical to YOLO26n (**2.57 M parameters, 24.4 GFLOPs**).

This repository is a self-contained fork of **Ultralytics 8.4.5** and contains everything needed to **reproduce every table in the paper from scratch**: environment, datasets, training, evaluation, the zero-cost deployment check, and edge-device benchmarking.


<img width="1450" height="760" alt="Fig" src="https://github.com/user-attachments/assets/9afb42f8-222c-499c-8b80-fa51f711df4e" />


---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [What This Repository Contains](#2-what-this-repository-contains)
3. [Installation](#3-installation)
4. [Datasets](#4-datasets)
5. [Pretrained & Trained Weights](#5-pretrained--trained-weights)
6. [Training](#6-training)
7. [Evaluation (Reproducing the Main Results)](#7-evaluation-reproducing-the-main-results)
8. [Zero-Cost Deployment & Edge Benchmark](#8-zero-cost-deployment--edge-benchmark)
9. [Inference on Images](#9-inference-on-images)
10. [Ablation Studies](#10-ablation-studies)
11. [Expected Results](#11-expected-results)
12. [Loss & Hyperparameter Reference](#12-loss--hyperparameter-reference)
13. [Repository Structure](#13-repository-structure)
14. [Troubleshooting](#14-troubleshooting)
15. [Citation](#15-citation)
16. [License, Contact & Acknowledgements](#16-license-contact--acknowledgements)

---

## 1. Quick Start

```bash
# 1. Clone and install this fork in editable mode 
git clone https://github.com/ashymuzuro/YOLO-SR.git
cd YOLO-SR
conda create -n yolosr python=3.12.3 -y && conda activate yolosr
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 2. Train YOLO-SR on VisDrone2019 from scratch (data auto-downloads on first run)
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR.yaml \
                        --data ultralytics/cfg/datasets/VisDrone.yaml \
                        --imgsz 1280 --epochs 300 --batch 4 --optimizer MuSGD --seed 0

# 3. Evaluate the trained model 
python val.py --weights runs/train/yolo_sr/weights/best.pt \
              --data ultralytics/cfg/datasets/VisDrone.yaml --imgsz 1280

# 4. Verify the zero-cost and export the deployment model 
python export_deploy.py --weights runs/train/yolo_sr/weights/best.pt --imgsz 1280 --format onnx
```




---

## 2. What This Repository Contains

All YOLO-SR additions follow the paper's single principle: auxiliary components run **only during training** and are removed or folded before deployment. The table maps each paper contribution to the exact file that implements it.

| Component | Paper | File | Summary |
|---|---|---|---|
| **RepCF** module | Sec. 3.3, Eq. 4–5 | `ultralytics/nn/modules/conv.py` | Training-time `3×3 + 1×1 + identity` fusion block; folds in closed form into a single `3×3` conv for deployment. Channel-preserving and purely linear so the fusion is exact. |
| **SAB** branch | Sec. 3.2, Eq. 3 | `ultralytics/cfg/models/26/YOLO-SR.yaml` (layers 25–28) | `Conv3×3 → Conv1×1(→3) → Upsample×8 → Sigmoid`, branching from backbone P3 (layer 4). Training-only. |
| **Structure-Aware Multi-Task Loss** | Sec. 3.4, Eq. 8–18 | `ultralytics/utils/loss.py` (`StructureAwareSRLoss`) | DASW-weighted pixel + Sobel-edge terms and a high-pass 2D-DFT frequency term; wired into the detection loss via `split_sr_output`. |
| Parser + fuse + deploy | Alg. 1 & 3 | `ultralytics/nn/tasks.py` | Registers `RepCF` (channel-preserving), fuses it in `model.fuse()`, and adds `BaseModel.strip_sr_branch()` to drop the SAB for a zero-parameter deployment checkpoint. |

The training forward pass returns `(detection_output, sr_reconstruction)`; at inference it returns at the `Detect` head (layer 24) and the SAB layers never execute, hence zero runtime cost. 

---

## 3. Installation

### Hardware used in the paper

| Component | Specification |
|---|---|
| CPU | Intel Xeon Platinum 8470Q (25 cores) |
| GPU | NVIDIA GeForce RTX 5090, 32 GB VRAM |
| OS | Ubuntu 22.04 LTS |

### Software Versions 

| Package | Version |
|---|---|
| Python | 3.12.3 |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Ultralytics  | 8.4.5 |

### Steps

```bash
git clone https://github.com/ashymuzuro/YOLO-SR.git
cd YOLO-SR

conda create -n yolosr python=3.12.3 -y
conda activate yolosr

# PyTorch built for CUDA 12.8
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the fork and custom RepCF / SAB / loss are registered
pip install -e .
```

Verify the install picked up the fork (not PyPI ultralytics):

```bash
python -c "from ultralytics.nn.modules import RepCF; print('RepCF OK')"
python -c "from ultralytics.utils.loss import StructureAwareSRLoss; print('SR loss OK')"
```

Both lines must print without error before you train.

---

## 4. Datasets

All three datasets are public. Each is trained/evaluated at **1280×1280** with the splits reported in the paper.

### 4.1 VisDrone2019-DET  Auto-Download

`ultralytics/cfg/datasets/VisDrone.yaml` includes a download hook that fetches the data and converts the annotations to YOLO format automatically on first use. No manual steps.

- Source: https://github.com/VisDrone/VisDrone-Dataset
- Splits: **train 6,471 / val 548 / test-dev 1,610** (test-challenge is not used)
- 10 classes: `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`

```bash
# Downloads to ../datasets/VisDrone on first train/val call  nothing to do manually
python val.py --weights yolo26n.pt --data ultralytics/cfg/datasets/VisDrone.yaml --imgsz 1280
```

### 4.2 TinyPerson

Config provided: **`ultralytics/cfg/datasets/TinyPerson.yaml`**.

- Source: https://github.com/ucas-vg/TinyBenchmark
- Splits: **train 794 / test 816**
- 2 classes: `earth_person, sea_person`

Download the dataset from the source above and arrange it in YOLO format:

```
datasets/TinyPerson/
├── images/{train,test}/*.jpg
└── labels/{train,test}/*.txt      # YOLO format: class cx cy w h  (normalised)
```

Then edit the `path:` field in `TinyPerson.yaml` if your dataset root differs.

### 4.3 HazyDet (Natural Haze Subset)

Config provided: **`ultralytics/cfg/datasets/HazyDet.yaml`**.

- Source: https://github.com/GrokCV/HazyDet
- Subset: **natural haze**, **train 8,000 / val 1,000 / test 2,000**
- 3 classes: `car, truck, bus`

Arrange as:

```
datasets/HazyDet/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt
```



---

## 5. Pretrained & Trained Weights

So users can reproduce accuracy **without re-training**, checkpoints are provided. The table states exactly where each file lives so nothing is assumed.

| File | Model | Dataset | mAP50 | Repository |
|---|---|---|---|:---:|
| `yolo-sr-visdrone.pt` | YOLO-SR (trained) | VisDrone2019 | 48.5 | Release download |
| `yolo26n-visdrone.pt` | YOLO26n baseline | VisDrone2019 | 44.8 | Release download |
| `yolo-sr-tinyperson.pt` | YOLO-SR | TinyPerson | 35.1 | Release download |
| `yolo-sr-hazydet.pt` | YOLO-SR | HazyDet | 71.4 | Release download |



Models: Download from Zenodo(https://zenodo.org/records/23655490)      DOI: 10.5281/zenodo.23655490
(https://doi.org/10.5281/zenodo.23655490)

---

## 6. Training

Training is **from scratch** (no pretrained weights) for every reported result, matching the paper. `train_yolo_sr.py` accepts all settings as flags; the defaults are the VisDrone configuration.

### 6.1 VisDrone2019 (300 epochs)

```bash
python train_yolo_sr.py \
  --model ultralytics/cfg/models/26/YOLO-SR.yaml \
  --data  ultralytics/cfg/datasets/VisDrone.yaml \
  --imgsz 1280 --epochs 300 --batch 4 --workers 8 \
  --optimizer MuSGD --close-mosaic 10 --seed 0 \
  --project runs/train --name yolo_sr_visdrone
```

### 6.2 HazyDet (300 epochs)

```bash
python train_yolo_sr.py \
  --model ultralytics/cfg/models/26/YOLO-SR.yaml \
  --data  ultralytics/cfg/datasets/HazyDet.yaml \
  --imgsz 1280 --epochs 300 --batch 4 --workers 8 \
  --optimizer MuSGD --close-mosaic 10 --seed 0 \
  --project runs/train --name yolo_sr_hazydet
```

### 6.3 TinyPerson (500 epochs, lrf 0.1)

```bash
python train_yolo_sr.py \
  --model ultralytics/cfg/models/26/YOLO-SR.yaml \
  --data  ultralytics/cfg/datasets/TinyPerson.yaml \
  --imgsz 1280 --epochs 500 --batch 4 --workers 8 \
  --optimizer MuSGD --close-mosaic 10 --lrf 0.1 --seed 0 \
  --project runs/train --name yolo_sr_tinyperson
```



---

## 7. Evaluation (Reproducing the Main Results)

`val.py` evaluates a checkpoint and prints Precision, Recall, mAP50, mAP50-95, parameter count and GFLOPs.

```bash
# VisDrone2019 (Table 2a)
python val.py --weights yolo-sr.pt --data ultralytics/cfg/datasets/VisDrone.yaml --imgsz 1280

# TinyPerson (Table 7)
python val.py --weights yolo-sr-tinyperson.pt --data ultralytics/cfg/datasets/TinyPerson.yaml --imgsz 1280

# HazyDet (Table 9)
python val.py --weights yolo-sr-hazydet.pt --data ultralytics/cfg/datasets/HazyDet.yaml --imgsz 1280
```

Add `--verbose` to print the **per-class AP50** breakdown (Table 3).

To evaluate the **YOLO26n baseline** for the direct comparison:

```bash
python val.py --weights yolo26n.pt --data ultralytics/cfg/datasets/VisDrone.yaml --imgsz 1280
```

---

## 8. Zero-Cost Deployment & Edge Benchmark

### 8.1 Re-parameterise and verify the footprint

`export_deploy.py` calls `strip_sr_branch()` (removes the SAB and folds every `RepCF` into one `3×3` conv), then `fuse()` (folds Conv+BN), prints the model complexity so you can confirm the zero-cost, and exports ONNX.

```bash
python export_deploy.py --weights runs/train/yolo_sr_visdrone/weights/best.pt \
                        --imgsz 1280 --format onnx
```



### 8.2 Edge Device : Huawei Atlas 200 DK (Table 6)

| Item | Specification |
|---|---|
| Board | Huawei Atlas 200 DK |
| Processor | Ascend 310 (8 TFLOPS FP16; 16/8/4 TOPS INT8) |
| Memory | 8 GB LPDDR4X, 3200 Mbps |
| Toolchain | CANN 5.0.2 |
| Runtime | ACL (Ascend Computing Language), Python 3.7 |

Pipeline:

```bash
# 1) On the training host: produce the deployable ONNX (Section 8.1)
python export_deploy.py --weights <best.pt> --imgsz 1280 --format onnx

# 2) In the CANN 5.0.2 environment: convert ONNX -> Ascend .om with ATC
atc --model=yolo-sr.onnx --framework=5 --output=yolo-sr \
    --input_format=NCHW --input_shape="images:1,3,1280,1280" \
    --soc_version=Ascend310

# 3) On the board: benchmark with the ACL runtime
#    Protocol used in the paper: 10 warm-up runs, then 100 timed runs, batch = 1.
#    Report latency (ms/image), FPS, and peak memory.
```




---

## 9. Inference on Images

```bash
# Single image or a folder of images
python detect.py --weights yolo-sr.pt --source path/to/images/ --imgsz 1280 --save
```

Results are written to `runs/detect/`.

---

## 10. Ablation Studies

The three components are `A = RepCF`, `B = Structure-Aware Multi-Task Loss`, `C = SAB`. This repository ships **ready-made config files and CLI flags** so each row of the paper's ablation tables is a single reproducible command, no hand-editing of layer indices. The full mapping (row → config + flags → expected mAP50) is in **[`ABLATIONS.md`](ABLATIONS.md)**; the paragraphs below summarise it.

### 10.1 Component ablation (Table 4)

Each structural variant is its own model YAML in `ultralytics/cfg/models/26/`:

| Config | A | B | C | Table 4 row | mAP50 |
|---|:---:|:---:|:---:|---|---|
| `yolo26n.yaml` | — | — | — | baseline | 44.8 |
| `YOLO-SR-noSAB.yaml` | ✓ | — | — | + A | 45.5 |
| `YOLO-SR-noRepCF.yaml` | — | ✓ | ✓ | + B + C | 48.1 |
| `YOLO-SR.yaml` | ✓ | ✓ | ✓ | full | 48.5 |

```bash
python train_yolo_sr.py --model ultralytics/cfg/models/26/yolo26n.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name abl_baseline
# ...see ABLATIONS.md for the other three commands
```

### 10.2 Mechanism-Isolation Ablation (Table 5)

Table 5 isolates each design choice. Two extra model configs support it `YOLO-SR-plain3x3.yaml` (RepCF replaced by a single 3×3 conv, the fair control for RepCF) and `YOLO-SR-P2.yaml` (a permanent stride-4 P2 head, the conventional small-object alternative to the SAB) and the SR-loss terms are toggled from the CLI:

| Flag | Symbol | Default | Effect |
|---|---|---|---|
| `--lambda-sr` | λ_sr | 0.1 | `0.0` disables the SR loss |
| `--gamma` | γ | 0.5 | edge-term weight |
| `--mu` | μ | 0.05 | `0.0` disables the frequency (FDC) term |
| `--eta` | η | 4.0 | `0.0` makes the SR loss uniform (DASW off) |

These flags set environment variables that `ultralytics/utils/loss.py` reads when it builds the criterion, so the paper defaults stay in force whenever a flag is omitted. Example: the "SAB + uniform SR loss" row (DASW and FDC off):

```bash
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noRepCF.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --eta 0 --mu 0 --name t5_sab_uniform
```

Run the **entire Table 5 sweep** with one command:

```bash
bash run_table5.sh
```

The full nine-row table (config + flags + expected Param/GFLOPs/mAP for every row, including the note on why the P2-head row is the only one whose footprint grows to 2.75 M / 31.2 GFLOPs) is in [`ABLATIONS.md`](ABLATIONS.md).



---

## 11. Expected Results

Reproduced numbers should match the following (from-scratch, 1280×1280).

### VisDrone2019 Controlled Comparison (Table 2a)

| Method | mAP50 | Precision | Recall | mAP50-95 | Param (M) | GFLOPs |
|---|---|---|---|---|---|---|
| YOLO26n (baseline) | 44.8 | 53.1 | 43.6 | 27.3 | 2.57 | 24.4 |
| **YOLO-SR (ours)** | **48.5** | **55.4** | **47.2** | **29.9** | **2.57** | **24.4** |

### TinyPerson (Table 7) & HazyDet (Table 9)

| Dataset | Model | mAP50 | Precision | Recall | mAP50-95 | Param (M) | GFLOPs |
|---|---|---|---|---|---|---|---|
| TinyPerson | YOLO26n | 28.5 | 44.2 | 27.4 | 10.2 | 2.57 | 24.4 |
| TinyPerson | **YOLO-SR** | **35.1** | **51.4** | **34.2** | **15.1** | **2.57** | **24.4** |
| HazyDet | YOLO26n | 68.8 | 79.5 | 61.5 | 48.1 | 2.57 | 24.4 |
| HazyDet | **YOLO-SR** | **71.4** | **82.5** | **65.8** | **51.2** | **2.57** | **24.4** |

### Mechanism-Isolation Ablation VisDrone2019 (Table 5)

| Variant | Param (M) | GFLOPs | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| YOLO26n baseline | 2.57 | 24.4 | 53.1 | 43.6 | 44.8 | 27.3 |
| SAB + uniform SR loss | 2.57 | 24.4 | 53.3 | 44.0 | 45.1 | 27.5 |
| SAB + DASW SR loss | 2.57 | 24.4 | 54.0 | 44.9 | 46.0 | 28.1 |
| SAB + DASW + FDC | 2.57 | 24.4 | 54.5 | 45.8 | 47.1 | 28.9 |
| Plain 3×3 fusion | 2.57 | 24.4 | 53.6 | 44.3 | 45.4 | 27.7 |
| RepCF fused | 2.57 | 24.4 | 54.1 | 44.8 | 46.1 | 28.2 |
| P2 detection head | 2.75 | 31.2 | 54.4 | 46.0 | 47.4 | 29.0 |
| P3-SAB branch | 2.57 | 24.4 | 54.8 | 46.6 | 48.0 | 29.4 |
| **Full YOLO-SR** | **2.57** | **24.4** | **55.4** | **47.2** | **48.5** | **29.9** |

### Edge Device Atlas 200 DK (Table 6)

| Model | Latency (ms) | FPS | Peak Memory (MB) | GFLOPs | Param (M) |
|---|---|---|---|---|---|
| YOLO26n (baseline) | 7.01 | 142.8 | 74.6 | 24.4 | 2.57 |
| **YOLO-SR (ours)** | 7.00 | 142.9 | 74.6 | 24.4 | 2.57 |

> Small variations (±0.1–0.3 mAP, minor FPS jitter) are normal across GPUs/driver versions even with a fixed seed, due to non-deterministic CUDA kernels. The relative gains over YOLO26n are the reproducible claim.

---

## 12. Loss & Hyperparameter Reference

### Training hyperparameters (Table 1)

Only three values differ across datasets.

| Hyperparameter | VisDrone2019 | HazyDet | TinyPerson |
|---|---|---|---|
| Epochs | 300 | 300 | **500** |
| Image size (px) | 1280 | 1280 | 1280 |
| Batch size | 4 | 4 | 4 |
| Workers | 8 | 8 | 8 |
| Optimizer | MuSGD | MuSGD | MuSGD |
| `lr0` | 0.01 | 0.01 | 0.01 |
| `lrf` | 0.01 | 0.01 | **0.1** |
| Momentum | 0.937 | 0.937 | 0.937 |
| Weight decay | 0.0005 | 0.0005 | 0.0005 |
| Warmup epochs | 3 | 3 | 3 |
| Warmup momentum | 0.8 | 0.8 | 0.8 |
| Warmup bias lr | 0.1 | 0.1 | 0.1 |
| `cos_lr` | False | False | False |
| Seed | 0 | 0 | 0 |

**Augmentation (identical across datasets).** Enabled: `auto_augment=randaugment`, `fliplr=0.5`, `translate=0.1`, `scale=0.5`, `mosaic=1.0`, `erasing=0.4`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`. Disabled to preserve UAV spatial integrity: `degrees=0.0`, `flipud=0.0`, `mixup=0.0`. (`close_mosaic=10` turns mosaic off for the final 10 epochs.)

### Loss Constants (Implemented in `ultralytics/utils/loss.py`)

```
L_total = L_det + lambda_sr * ( L_pixel^DASW + gamma * L_edge^DASW + mu * L_freq )   # Eq. 6/17/18
lambda_sr = 0.1   gamma = 0.5   mu = 0.05                                            # Eq. 17
DASW:  M(x,y) = 1 + eta * sum_k 1[(x,y) in b_k] * A_max / (A_k + eps)                # Eq. 15
       eta = 4.0   A_max = 1024 (32x32)   eps = 1
FDC high-pass radius:  r_lo = min(H, W) / 8                                          # Eq. 13
```

`lambda_sr`, `gamma`, `mu`, `eta` default to these values and can be overridden per run with the `--lambda-sr / --gamma / --mu / --eta` flags of `train_yolo_sr.py` (used for the Table 5 ablation; see [`ABLATIONS.md`](ABLATIONS.md)).

---

## 13. Repository Structure

```
YOLO-SR/
├── README.md                         # this file
├── ABLATIONS.md                      # Table 4 / Table 5 reproduction guide
├── run_table5.sh                     # one-command Table 5 sweep
├── train_yolo_sr.py                  # training entry point (argparse + loss flags)
├── val.py                            # evaluation entry point
├── detect.py                         # image/folder inference (argparse)
├── export_deploy.py                  # strip SAB + fuse RepCF + verify + export ONNX
├── requirements.txt
├── pyproject.toml                    # installs this fork as `ultralytics` (editable)
├── LICENSE
│
└── ultralytics/                      # modified Ultralytics 8.4.5
    ├── cfg/
    │   ├── models/26/
    │   │   ├── YOLO-SR.yaml           # full YOLO-SR training graph (SAB + RepCF)
    │   │   ├── yolo26n.yaml           # baseline (ablation)
    │   │   ├── YOLO-SR-noSAB.yaml     # RepCF only (ablation)
    │   │   ├── YOLO-SR-noRepCF.yaml   # SAB + loss (ablation)
    │   │   ├── YOLO-SR-plain3x3.yaml  # plain 3x3 fusion (Table 5)
    │   │   └── YOLO-SR-P2.yaml        # permanent P2 head (Table 5)
    │   └── datasets/
    │       ├── VisDrone.yaml          # auto-download + convert
    │       ├── TinyPerson.yaml
    │       └── HazyDet.yaml
    ├── nn/
    │   ├── modules/conv.py            # RepCF module
    │   └── tasks.py                   # RepCF parsing/fusion + strip_sr_branch()
    └── utils/loss.py                  # StructureAwareSRLoss (+ env-var overrides)
```

---

## 14. Troubleshooting

- **`RepCF` / `StructureAwareSRLoss` import fails** → you are on the PyPI `ultralytics`, not this fork. Reinstall from the repo root with `pip install -e .`.
- **CUDA / PyTorch mismatch** → install the CUDA 12.8 build: `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`.
- **`MuSGD` optimizer not found** → it ships with this fork; confirm you installed the repo (`pip install -e .`), not PyPI.
- **Out of memory at `--imgsz 1280 --batch 4`** → the paper used a 32 GB RTX 5090. Reduce `--batch`; note that changing batch size can slightly shift results.
- **Params/GFLOPs not 2.57 M / 24.4** → make sure `export_deploy.py` ran `strip_sr_branch()` **and** `fuse()`; the check must be on the stripped+fused model. (The P2-head ablation is the one intended exception at 2.75 M / 31.2 GFLOPs.)
- **Ablation loss flags seem ignored** → they only affect training when the SAB is present (the loss supervises the SAB output). Use them with an SAB config (`YOLO-SR.yaml` or `YOLO-SR-noRepCF.yaml`); the training log prints the active override values.
- **VisDrone did not download** → check network access; the download hook lives in `ultralytics/cfg/datasets/VisDrone.yaml`.

---

## 15. Citation

```bibtex
@article{muzuro_yolosr,
  title   = {A Zero-Cost Super-Resolution and Edge-Aware Architecture for Micro-Scale UAV Object Detection},
  author  = {Muzuro, Ashley Tendai and Tian, Junwei and Lambou Lado, Aude Blanche and Geinore, Teshale Woldetinsae and Ngalati, Dalitso},
  journal = {Pattern Analysis and Applications},
  year    = {2025},
  note    = {Under review}
}
```

Update with the final volume/DOI on acceptance.

---

## 16. License, Contact & Acknowledgements

**License.** This project inherits the **AGPL-3.0** license [`LICENSE`](LICENSE).

**Contact.** Ashley Tendai Muzuro : ashley.muzuro@yahoo.com

**Acknowledgements.** [Ultralytics](https://github.com/ultralytics/ultralytics). Datasets: [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), [TinyPerson](https://github.com/ucas-vg/TinyBenchmark), [HazyDet](https://github.com/GrokCV/HazyDet).
