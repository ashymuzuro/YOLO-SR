#!/usr/bin/env bash
# Reproduce the mechanism-isolation ablation (Table 5) on VisDrone2019.
# Each row is trained from scratch with the paper's settings; only the model
# config and the SR-loss flags change between rows. See ABLATIONS.md for the
# row -> config/flags mapping and the expected mAP50 values.
#
# Usage:
#   bash run_table5.sh                 # train every row sequentially
#   DATA=ultralytics/cfg/datasets/VisDrone.yaml EPOCHS=300 bash run_table5.sh
#
# Override any of these via environment variables:
DATA="${DATA:-ultralytics/cfg/datasets/VisDrone.yaml}"
IMGSZ="${IMGSZ:-1280}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-4}"
DEVICE="${DEVICE:-0}"
M=ultralytics/cfg/models/26
COMMON="--data $DATA --imgsz $IMGSZ --epochs $EPOCHS --batch $BATCH --device $DEVICE --seed 0"

set -e

echo "[Table 5] Row 1/9  YOLO26n baseline"
python train_yolo_sr.py --model $M/yolo26n.yaml         $COMMON --name t5_baseline

echo "[Table 5] Row 2/9  SAB + uniform SR loss (DASW off, FDC off)"
python train_yolo_sr.py --model $M/YOLO-SR-noRepCF.yaml $COMMON --eta 0 --mu 0 --name t5_sab_uniform

echo "[Table 5] Row 3/9  SAB + DASW SR loss (FDC off)"
python train_yolo_sr.py --model $M/YOLO-SR-noRepCF.yaml $COMMON --eta 4 --mu 0 --name t5_sab_dasw

echo "[Table 5] Row 4/9  SAB + DASW + FDC (full SR loss)"
python train_yolo_sr.py --model $M/YOLO-SR-noRepCF.yaml $COMMON --name t5_sab_dasw_fdc

echo "[Table 5] Row 5/9  Plain 3x3 fusion"
python train_yolo_sr.py --model $M/YOLO-SR-plain3x3.yaml $COMMON --name t5_plain3x3

echo "[Table 5] Row 6/9  RepCF fused"
python train_yolo_sr.py --model $M/YOLO-SR-noSAB.yaml   $COMMON --name t5_repcf

echo "[Table 5] Row 7/9  P2 detection head (permanent; 2.75 M / 31.2 GFLOPs)"
python train_yolo_sr.py --model $M/YOLO-SR-P2.yaml      $COMMON --name t5_p2_head

echo "[Table 5] Row 8/9  P3-SAB branch (zero-cost alternative to P2)"
python train_yolo_sr.py --model $M/YOLO-SR-noRepCF.yaml $COMMON --name t5_p3_sab

echo "[Table 5] Row 9/9  Full YOLO-SR"
python train_yolo_sr.py --model $M/YOLO-SR.yaml         $COMMON --name t5_full

echo ""
echo "All Table 5 rows trained. Evaluate each with:"
echo "  python val.py --weights runs/train/<name>/weights/best.pt --data $DATA --imgsz $IMGSZ"
