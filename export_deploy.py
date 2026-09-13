# -*- coding: utf-8 -*-
"""YOLO-SR deployment / export helper.

Produces the zero-cost deployment graph:
  1) strip_sr_branch() removes the training-only SAB layers and fuses every
     RepCF block into a single equivalent 3x3 convolution;
  2) fuse() folds Conv+BN across the rest of the network;
  3) export() writes the deployable model (e.g. ONNX).
"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/train/yolo_sr/weights/best.pt")  # your trained checkpoint

    # Remove the Super-Resolution Auxiliary Branch and fold RepCF -> single 3x3.
    # (Safe: the SAB layers are the trailing modules and nothing references them.)
    model.model.strip_sr_branch()
    model.model.fuse()

    # Sanity check: the SAB layers should be gone and RepCF fused.
    print(model.model)

    model.export(format="onnx", imgsz=1280, opset=12)
