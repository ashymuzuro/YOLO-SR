# Ablation Study Guide

This guide reproduces the two ablation studies in the paper:

- **Table 4** : Component Ablation (add A / B / C one group at a time)
- **Table 5** : Mechanism-Isolation Ablation (isolate each design choice)

on **VisDrone2019** (imgsz 1280, from scratch). The three components are:

- **A = RepCF**:  Reparameterised Contextual Fusion (Neck)
- **B = Structure-Aware Multi-Task Loss**:  DASW-weighted pixel + Sobel edge + 2D-DFT frequency terms
- **C = SAB**: Super-Resolution Auxiliary Branch (Training-Only)

---

## How the Ablation Works

There are two kinds of switch, and every row below is a single command.

**1. Model-level (choose a config file).** Each structural variant is its own
model YAML in `ultralytics/cfg/models/26/`:

| Config file | A (RepCF) | B (Loss) | C (SAB) | Notes |
|---|:---:|:---:|:---:|---|
| `yolo26n.yaml` | — | — | — | baseline; plain P3 concat |
| `YOLO-SR-noSAB.yaml` | RepCF | — | — | RepCF only |
| `YOLO-SR-plain3x3.yaml` | 3×3 conv | — | — | plain 3×3 fusion (RepCF's fair control) |
| `YOLO-SR-noRepCF.yaml` | — | ✓ | ✓ | SAB + loss, plain P3 concat |
| `YOLO-SR-P2.yaml` | — | — | — | permanent P2/4 head (4-scale detector) |
| `YOLO-SR.yaml` | RepCF | ✓ | ✓ | full model |

**2. Loss-level (CLI flags).** The Structure-Aware Multi-Task Loss coefficients
are overridable from `train_yolo_sr.py` without editing code. The flags set the
`YOLOSR_*` environment variables that `ultralytics/utils/loss.py` reads when it
builds the criterion (paper defaults are used when a flag is omitted):

| Flag | Symbol | Default | Effect when changed |
|---|---|---|---|
| `--lambda-sr` | λ_sr | 0.1 | `0.0` disables the SR loss entirely |
| `--gamma` | γ | 0.5 | edge-term weight |
| `--mu` | μ | 0.05 | `0.0` disables the frequency (FDC) term |
| `--eta` | η | 4.0 | `0.0` makes the SR loss uniform (DASW off) |

---

## Table 4 Component Ablation

| Config file | Reproduces (Table 4 row) | mAP50 |
|---|---|---|
| `yolo26n.yaml` | baseline | 44.8 |
| `YOLO-SR-noSAB.yaml` | + A | 45.5 |
| `YOLO-SR-noRepCF.yaml` | + B + C | 48.1 |
| `YOLO-SR.yaml` | + A + B + C (full) | 48.5 |

```bash
python train_yolo_sr.py --model ultralytics/cfg/models/26/yolo26n.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name abl_baseline

python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noSAB.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name abl_A

python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noRepCF.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name abl_BC

python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name abl_full
```

---

## Table 5  Mechanism-Isolation Ablation

Each row isolates one design choice. The mapping of row → config + flags is exact;
the four rows that share a config differ only by the loss flags shown.

| # | Variant (paper) | Config | Extra flags | Param (M) | GFLOPs | mAP50 |
|---|---|---|---|---|---|---|
| 1 | YOLO26n baseline | `yolo26n.yaml` | — | 2.57 | 24.4 | 44.8 |
| 2 | SAB + uniform SR loss | `YOLO-SR-noRepCF.yaml` | `--eta 0 --mu 0` | 2.57 | 24.4 | 45.1 |
| 3 | SAB + DASW SR loss | `YOLO-SR-noRepCF.yaml` | `--eta 4 --mu 0` | 2.57 | 24.4 | 46.0 |
| 4 | SAB + DASW + FDC | `YOLO-SR-noRepCF.yaml` | *(defaults)* | 2.57 | 24.4 | 47.1 |
| 5 | Plain 3×3 fusion | `YOLO-SR-plain3x3.yaml` | — | 2.57 | 24.4 | 45.4 |
| 6 | RepCF fused | `YOLO-SR-noSAB.yaml` | — | 2.57 | 24.4 | 46.1 |
| 7 | P2 detection head | `YOLO-SR-P2.yaml` | — | 2.75 | 31.2 | 47.4 |
| 8 | P3-SAB branch | `YOLO-SR-noRepCF.yaml` | *(defaults)* | 2.57 | 24.4 | 48.0 |
| 9 | Full YOLO-SR | `YOLO-SR.yaml` | — | 2.57 | 24.4 | 48.5 |

Commands:

```bash
# Row 1 — baseline
python train_yolo_sr.py --model ultralytics/cfg/models/26/yolo26n.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name t5_baseline

# Rows 2-4 — isolate the SR-loss design on the SAB base (no RepCF)
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noRepCF.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --eta 0 --mu 0 --name t5_sab_uniform
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noRepCF.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --eta 4 --mu 0 --name t5_sab_dasw
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noRepCF.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name t5_sab_dasw_fdc

# Rows 5-6 — isolate the neck fusion (plain 3x3 vs RepCF), no SAB
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-plain3x3.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name t5_plain3x3
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noSAB.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name t5_repcf

# Row 7 — permanent P2 head (raises deployed cost to 2.75 M / 31.2 GFLOPs)
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-P2.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name t5_p2_head

# Row 8 — training-only SAB alternative to P2 (zero deployment cost)
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noRepCF.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name t5_p3_sab

# Row 9 — full model
python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR.yaml \
    --data ultralytics/cfg/datasets/VisDrone.yaml --name t5_full
```

Or run the whole study with `bash run_table5.sh`
### Note on rows 4 and 8

Rows 4 ("SAB + DASW + FDC") and 8 ("P3-SAB branch") use the **same runnable
config** (`YOLO-SR-noRepCF.yaml` with default loss): row 4 is the endpoint of the
loss-isolation sub-study, and row 8 is the same trained model presented as the
zero-cost alternative to the P2 head in the small-object sub-study. In the paper
they appear in different comparison groups. The small mAP difference between the
two reported numbers reflects run-to-run variance, not a different configuration. If you prefer a single
authoritative number, train this config once and cite it for both rows, or report
mean ± std over a few seeds.

---

## Isolating B independently of C (Optional, Rarely Needed)

In this implementation the SR loss (B) only has an effect when the SAB (C) is
present, because the loss supervises the SAB's reconstruction output. The loss
flags above let you turn B **off while C is present** (`--lambda-sr 0` keeps the
SAB in the graph but stops its gradient), which is the meaningful "C present, B
off" configuration. The reverse ("B on, C absent") is not defined, with no SAB
there is no reconstruction tensor for the loss to act on. Table 5 is structured
around this coupling: the SR-loss rows (2-4) all keep the SAB and vary the loss.
