# -*- coding: utf-8 -*-
"""YOLO-SR training entry point.

Trains the YOLO-SR model (YOLO26n baseline + training-only Super-Resolution
Auxiliary Branch, RepCF neck fusion, and the Structure-Aware Multi-Task Loss).
The auxiliary branch and RepCF multi-branch topology are active only during
training; they are folded / pruned for deployment (see export_deploy.py).
"""
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO

if __name__ == "__main__":
    # Build YOLO-SR from the annotated config (do NOT load pretrained weights if
    # you want the paper's from-random-initialisation protocol).
    model = YOLO("ultralytics/cfg/models/26/YOLO-SR.yaml")

    model.train(
        data="ultralytics/cfg/datasets/VisDrone.yaml",  # adjust to your dataset yaml
        imgsz=1280,      # UAV small-object resolution used in the paper
        epochs=300,
        batch=4,
        workers=8,
        device=0,
        optimizer="MuSGD",
        close_mosaic=10,
        project="runs/train",
        name="yolo_sr",
    )
