# -*- coding: utf-8 -*-
"""YOLO-SR deployment / export helper.

Produces the zero-cost deployment graph and verifies the footprint claim:

  1) strip_sr_branch() removes the training-only Super-Resolution Auxiliary
     Branch (SAB) and fuses every RepCF block into a single equivalent 3x3 conv;
  2) fuse() folds Conv+BN across the rest of the network;
  3) prints parameter count and GFLOPs so the 2.57 M / 24.4 GFLOPs claim
     (identical to YOLO26n, Table 6) can be confirmed;
  4) export() writes the deployable model (ONNX by default).

    python export_deploy.py --weights runs/train/yolo_sr/weights/best.pt \
        --imgsz 1280 --format onnx
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Strip + fuse + export YOLO-SR for deployment")
    p.add_argument("--weights", required=True, help="trained .pt checkpoint")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--format", default="onnx", help="export format (onnx, torchscript, ...)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset")
    p.add_argument("--no-export", action="store_true",
                   help="only strip/fuse and report complexity, do not export a file")
    return p.parse_args()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()

    model = YOLO(args.weights)

    print("===== Before re-parameterisation (training graph) =====")
    print(f"parameters: {n_params(model.model) / 1e6:.3f} M")

    # Remove the Super-Resolution Auxiliary Branch and fold RepCF -> single 3x3.
    model.model.strip_sr_branch()
    model.model.fuse()

    print("\n===== After re-parameterisation (deployment graph) =====")
    model.info(verbose=False)  # prints layers, parameters, gradients, GFLOPs at --imgsz
    print(f"parameters: {n_params(model.model) / 1e6:.3f} M")
    print("Expected: 2.57 M parameters, 24.4 GFLOPs (identical to YOLO26n).")

    if not args.no_export:
        path = model.export(format=args.format, imgsz=args.imgsz, opset=args.opset)
        print(f"\nExported deployment model -> {path}")


if __name__ == "__main__":
    main()
