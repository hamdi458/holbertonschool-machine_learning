#!/usr/bin/env python3
"""class Yolo"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line[0:-1] for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, X):
        """ sigmoid function"""
        return (1 / (1 + np.exp(-X)))

    def process_outputs(self, outputs, image_size):
        """
        ARGS:
        outputs is a list of numpy.ndarrays containing :
        (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
        *grid_height & grid_width => the height and width of the grid
        *anchor_boxes => the number of anchor boxes used
        *4 => (t_x, t_y, t_w, t_h)
        *1 => box_confidence
        *classes => class probabilities for all classes
        """
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

            grid_height, grid_width, anchor_boxes, _ = out.shape
            """ grid idices"""
            cx = np.indices((grid_height, grid_height, anchor_boxes))[1]
            cy = np.indices((grid_height, grid_height, anchor_boxes))[0]
            """ localisation in grid """
            bx = (self.sigmoid(t_x) + cx)
            by = (self.sigmoid(t_y) + cy)
            """ localisation in images of shape [13x13,26x26,525,52]"""
            bx = bx / grid_width
            by = by / grid_height

            """ from list of anchors
            anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
            """
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            """ localisation in image of the model input"""
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

            bw = pw * np.exp(t_w) / input_w
            bh = ph * np.exp(t_h) / input_h
            """ rescale coordinates to original dimensions"""
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
