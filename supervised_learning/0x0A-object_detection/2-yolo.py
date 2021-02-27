#!/usr/bin/env python3
"""class Yolo"""
import tensorflow.keras as K
import numpy as np


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

    def sigmoid(self, X):
        """function actication sigmoid"""
        return (1 / (1 + np.exp(-X)))

    def process_outputs(self, outputs, image_size):
        """grrrrrrrrr"""
        i = 0
        boxes = []
        image_height, image_width = image_size
        box_confidence = []
        box_class_probs = []
        for out in outputs:
            boxes.append(out[:, :, :, 0:4])
            box_confidence.append(self.sigmoid(out[:, :, :, 4:5]))
            box_class_probs.append(self.sigmoid(out[:, :, :, 5:]))
            t_x = boxes[i][:, :, :, 0]
            t_y = boxes[i][:, :, :, 1]
            t_w = boxes[i][:, :, :, 2]
            t_h = boxes[i][:, :, :, 3]
            grid_height = out.shape[0]
            grid_width = out.shape[1]
            anchor_boxes = out.shape[2]
            """#indices off grid"""
            cx = np.indices((grid_height, grid_width, anchor_boxes))[1]
            cy = np.indices((grid_height, grid_width, anchor_boxes))[0]
            """#indices of tx in grid"""
            bx = self.sigmoid(t_x) + cx
            """#pos in image"""
            bx = bx / grid_width
            by = self.sigmoid(t_y) + cy
            by = by = by / grid_height
            """#anchor shape"""
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            """"#input model shape"""
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

            bw = pw * np.exp(t_w) / input_w
            bh = ph * np.exp(t_h) / input_h

            x1 = (bx - bw / 2) * image_width
            x2 = (bx - bw / 2 + bw) * image_width
            y1 = (by - bh / 2) * image_height
            y2 = (by - bh / 2 + bh) * image_height

            boxes[i][:, :, :, 0] = x1
            boxes[i][:, :, :, 1] = y1
            boxes[i][:, :, :, 2] = x2
            boxes[i][:, :, :, 3] = y2
            i = i + 1

        return boxes, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Returns a tuple of (filtered_boxes, box_classes, box_scores)"""
        scores = []
        box_classes = []
        box_class_scores = []
        filtering_mask = []
        filtred_boxes = []
        classes = []
        box_scores = []
        for i in range(len(boxes)):
            box_scores.append(np.multiply(box_confidences[i],
                                          box_class_probs[i]))
            box_classes.append(np.argmax(box_scores[i], axis=3))
            box_class_scores.append(np.max(box_scores[i], axis=-1))
            filtering_mask.append(box_class_scores[i] >= self.class_t)

        filtred_boxes += (d[s] for d, s in zip(boxes, filtering_mask))
        scores += (d[s] for d, s in zip(box_class_scores, filtering_mask))

        classes += (d[s].flatten() for d, s in zip(box_classes,
                                                   filtering_mask))

        classes = np.concatenate(classes).ravel()
        filtred_boxes = np.concatenate(filtred_boxes)
        scores = np.concatenate(scores).ravel()
        return (filtred_boxes,  classes, scores)
