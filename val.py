# -*- coding: utf-8 -*-
"""YOLO-SR evaluation entry point.

Evaluates a trained checkpoint and prints Precision, Recall, mAP50, mAP50-95,
parameter count and GFLOPs. Reproduces the paper's results tables:

    # VisDrone2019 (Table 2a); add --verbose for per-class AP50 (Table 3)
    python val.py --weights yolo-sr.pt --data ultralytics/cfg/datasets/VisDrone.yaml --imgsz 1280

    # TinyPerson (Table 7)
    python val.py --weights yolo-sr-tinyperson.pt --data ultralytics/cfg/datasets/TinyPerson.yaml --imgsz 1280

    # HazyDet (Table 9)
    python val.py --weights yolo-sr-hazydet.pt --data ultralytics/cfg/datasets/HazyDet.yaml --imgsz 1280

The Super-Resolution Auxiliary Branch does not execute at inference (the forward
pass returns at the Detect head), so validating the training checkpoint gives the
same detection accuracy as the stripped deployment model. Pass --verify-deploy to
additionally strip the SAB, fuse RepCF, and print the deployed complexity
(should read 2.57 M params / 24.4 GFLOPs, matching YOLO26n - the zero-cost claim).
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO-SR")
    p.add_argument("--weights", required=True, help="trained .pt checkpoint")
    p.add_argument("--data", required=True, help="dataset yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="dataset split to evaluate")
    p.add_argument("--verbose", action="store_true", help="print per-class AP50")
    p.add_argument("--verify-deploy", action="store_true",
                   help="after eval, strip SAB + fuse RepCF and print deployed params/GFLOPs")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        verbose=args.verbose,
    )

    box = metrics.box
    print("\n===== YOLO-SR evaluation =====")
    print(f"weights : {args.weights}")
    print(f"data    : {args.data}  (split={args.split})")
    print(f"Precision : {box.mp:.4f}")
    print(f"Recall    : {box.mr:.4f}")
    print(f"mAP50     : {box.map50:.4f}")
    print(f"mAP50-95  : {box.map:.4f}")

    if args.verify_deploy:
        print("\n===== Zero-cost deployment check =====")
        model.model.strip_sr_branch()  # remove SAB, fold every RepCF -> one 3x3 conv
        model.model.fuse()             # fold Conv+BN
        model.info(verbose=False)      # prints layers, parameters, gradients, GFLOPs
        print("Expected: 2.57 M parameters, 24.4 GFLOPs (identical to YOLO26n).")


if __name__ == "__main__":
    main()
