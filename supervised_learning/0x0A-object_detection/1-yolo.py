#!/usr/bin/env python3
"""class Yolo"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initialize"""
        self.model = K.models.load_model(filepath=model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.class_names = []
        self.anchors = anchors
        with open(classes_path, 'r') as f:
            for ligne in f:
                self.class_names.append(ligne[0: -1])

    def sigmoid(self, X):
        """function actication sigmoid"""
        return (1 / (1 + np.exp(-X)))

    def process_outputs(self, outputs, image_size):
        """outputs process"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        ih, iw = image_size
        for i, op in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, cls = op.shape
            box = np.zeros(op[..., :4].shape)

            t_x = op[..., 0]
            t_y = op[..., 1]
            t_w = op[..., 2]
            t_h = op[..., 3]

            # Calculate anchor boxes

            anchors_w = self.anchors[..., 0]
            # repeating each anchor belong all grids_w
            anchor_w = np.tile(anchors_w[i], grid_w)
            anchor_w = anchor_w.reshape(grid_w, 1, len(anchors_w[i]))

            anchors_h = self.anchors[..., 1]
            # repeating each anchor belong all grids_h
            anchor_h = np.tile(anchors_h[i], grid_h)
            anchor_h = anchor_h.reshape(grid_h, 1, len(anchors_h[i]))

            # Calculate corners
            cx = np.tile(np.arange(grid_w), grid_h)
            cx = cx.reshape(grid_w, grid_w, 1)
            cy = np.tile(np.arange(grid_h), grid_h)
            cy = cy.reshape(grid_h, grid_h).T
            cy = cy.reshape(grid_h, grid_h, 1)

            # prediction of each coordinate
            prediction_x = (1 / (1 + np.exp(-t_x))) + cx
            prediction_y = (1 / (1 + np.exp(-t_y))) + cy
            prediction_w = np.exp(t_w) * anchor_w
            prediction_h = np.exp(t_h) * anchor_h

            # Normalize values
            prediction_x /= grid_w
            prediction_y /= grid_h
            prediction_w /= self.model.input.shape[1]
            prediction_h /= self.model.input.shape[2]

            x1 = (prediction_x - (prediction_w / 2)) * iw
            y1 = (prediction_y - (prediction_h / 2)) * ih
            x2 = (prediction_x + (prediction_w / 2)) * iw
            y2 = (prediction_y + (prediction_h / 2)) * ih

            # Setting coordinates
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

            # Predict and set confidence
            confidence = (1 / (1 + np.exp(-op[..., 4])))
            confidence = confidence.reshape(grid_h, grid_w, anchor_boxes, 1)
            box_confidences.append(confidence)

            # Predict class probability
            prob = (1 / (1 + np.exp(-op[..., 5:])))
            box_class_probs.append(prob)

        return (boxes, box_confidences, box_class_probs)