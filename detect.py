# -*- coding: utf-8 -*-
"""
@Auth ： Ashley T Muzuro
@File ：detect.py
@IDE ：PyCharm
@Email ：ashley.muzuro@yahoo.com
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'C:\YOLO-SR\runs\detect\weights\best.pt')
    model.predict(source=r'C:\YOLO-SR\runs\detect\weights\test2,jpg', save=True, show=True,)
    
    