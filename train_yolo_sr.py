# -*- coding: utf-8 -*-
"""YOLO-SR training entry point.

Trains YOLO-SR (YOLO26n baseline + training-only Super-Resolution Auxiliary
Branch, RepCF neck fusion, and the Structure-Aware Multi-Task Loss). The
auxiliary branch and the RepCF multi-branch topology are active ONLY during
training; they are folded / pruned for deployment (see export_deploy.py).

All hyperparameters default to the values reported in the paper (Table 1 and the
augmentation list in Sec. 4.1.2), so the headline results reproduce with:

    # VisDrone2019 (300 epochs)
    python train_yolo_sr.py --data ultralytics/cfg/datasets/VisDrone.yaml \
        --name yolo_sr_visdrone

    # HazyDet (300 epochs)
    python train_yolo_sr.py --data ultralytics/cfg/datasets/HazyDet.yaml \
        --name yolo_sr_hazydet

    # TinyPerson (500 epochs, lrf 0.1)
    python train_yolo_sr.py --data ultralytics/cfg/datasets/TinyPerson.yaml \
        --epochs 500 --lrf 0.1 --name yolo_sr_tinyperson

Training is from scratch (no pretrained weights) to match the paper's protocol.
Pass --pretrained yolo26n.pt only if you deliberately want warm-start training.

------------------------------------------------------------------------------
Mechanism-isolation ablation (Table 5)
------------------------------------------------------------------------------
The Structure-Aware Multi-Task Loss coefficients can be overridden from the CLI
so each ablation row is a single reproducible command. The flags below set the
YOLOSR_* environment variables that ultralytics/utils/loss.py reads when it
constructs the SR loss (paper defaults are used when a flag is omitted):

    --gamma       edge-term weight        (default 0.5)
    --mu          frequency-term weight   (default 0.05; use 0.0 to disable FDC)
    --eta         DASW emphasis           (default 4.0; use 0.0 for uniform SR loss)
    --lambda-sr   det-vs-reconstruction   (default 0.1; use 0.0 to disable SR loss)

Example (SAB + uniform SR loss row: DASW off, frequency off):

    python train_yolo_sr.py --model ultralytics/cfg/models/26/YOLO-SR-noRepCF.yaml \
        --data ultralytics/cfg/datasets/VisDrone.yaml \
        --eta 0 --mu 0 --name t5_sab_uniform

See ABLATIONS.md for the full Table 4 / Table 5 command list.
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO-SR")
    p.add_argument("--model", default="ultralytics/cfg/models/26/YOLO-SR.yaml",
                   help="model config (YOLO-SR training graph)")
    p.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml",
                   help="dataset yaml")
    p.add_argument("--pretrained", default=None,
                   help="optional .pt to warm-start from (omit for the paper's from-scratch protocol)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--optimizer", default="MuSGD")
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01, help="final LR factor (paper: 0.01; TinyPerson: 0.1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo_sr")
    # --- Structure-Aware Multi-Task Loss overrides (Table 5 ablations) ---
    p.add_argument("--lambda-sr", type=float, default=None,
                   help="SR-loss weight (default 0.1; 0.0 disables the SR loss)")
    p.add_argument("--gamma", type=float, default=None,
                   help="edge-term weight (default 0.5)")
    p.add_argument("--mu", type=float, default=None,
                   help="frequency-term weight (default 0.05; 0.0 disables FDC)")
    p.add_argument("--eta", type=float, default=None,
                   help="DASW emphasis (default 4.0; 0.0 gives a uniform SR loss)")
    return p.parse_args()


def main():
    args = parse_args()

    # Plumb the SR-loss overrides through to loss.py via environment variables
    # BEFORE the model / criterion is constructed. Only set the ones the user
    # provided so the paper defaults remain in force otherwise.
    if args.lambda_sr is not None:
        os.environ["YOLOSR_LAMBDA_SR"] = str(args.lambda_sr)
    if args.gamma is not None:
        os.environ["YOLOSR_GAMMA"] = str(args.gamma)
    if args.mu is not None:
        os.environ["YOLOSR_MU"] = str(args.mu)
    if args.eta is not None:
        os.environ["YOLOSR_ETA"] = str(args.eta)

    # Import after the env vars are set (loss.py reads them at criterion build time).
    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.pretrained:
        model.load(args.pretrained)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        project=args.project,
        name=args.name,
        # --- Optimiser schedule (Table 1) ---
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=False,
        # --- Data augmentation (Sec. 4.1.2) ---
        auto_augment="randaugment",
        fliplr=0.5,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Orientation-/structure-breaking augmentations disabled for UAV imagery
        degrees=0.0,
        flipud=0.0,
        mixup=0.0,
    )


if __name__ == "__main__":
    main()
