#!/usr/bin/env python3
"""Initialize Yolo"""


import tensorflow.keras as K
import numpy as np


class Yolo:
    """class Yolo that uses the Yolo v3 algorithm to perform object detectio"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor.
        model_path: is the path to where a Darknet Keras model is stored
        classes_path: is the path to where the list of class names used for the
        Darknet model, listed in order of index, can be found
        class_t: is a float representing the box score threshold for the
        initial filtering step
        nms_t: is a float representing the IOU threshold for non-max
        suppression
        anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
            outputs: is the number of outputs (predictions) made by the Darknet
            model
            anchor_boxes: is the number of anchor boxes used for each
            prediction 2 => [anchor_box_width, anchor_box_height]"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.readlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Function that process outputs
        outputs is a list of numpy.ndarrays containing the predictions from
        the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of the grid
                used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original size
        [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively:
            4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative to
                original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences for
            each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box’s class
            probabilities for each output, respectively"""
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
