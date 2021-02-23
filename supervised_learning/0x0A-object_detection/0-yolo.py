#!/usr/bin/env python3
"""class Yolo"""
import tensorflow.keras as K


class Yolo:
    """uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(filepath=model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.class_names = []
        self.anchors = anchors
        with open(classes_path, 'r') as f:
            for ligne in f:
                self.class_names.append(ligne[0: -1])
