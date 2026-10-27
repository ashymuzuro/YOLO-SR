# -*- coding: utf-8 -*-
"""YOLO-SR inference on images.

Runs detection on a single image or a folder of images and saves the annotated
results to runs/detect/.

    python detect.py --weights yolo-sr.pt --source path/to/images/ --imgsz 1280 --save

@Author : Ashley T. Muzuro
@Email  : ashley.muzuro@yahoo.com
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-SR inference")
    p.add_argument("--weights", required=True, help="trained .pt checkpoint (e.g. yolo-sr.pt)")
    p.add_argument("--source", required=True, help="image file, folder, or glob")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--device", default="0")
    p.add_argument("--save", action="store_true", help="save annotated images")
    p.add_argument("--show", action="store_true", help="display results (desktop only)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()
