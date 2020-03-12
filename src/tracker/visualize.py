import cv2
import torch
import numpy as np


class IDVisualizer(object):
    def __init__(self, with_bbox=True):
        self.with_bbox = with_bbox
        self.boxid_color = (255, 255, 0)
        self.box_color = (0, 0, 255)
        self.skeid_color = (0, 255, 255)

    def plot_bbox_id(self, id2bbox, img):
        for idx in range(len(id2bbox)):
            [x1, y1, x2, y2] = list(id2bbox.values())[idx]
            cv2.putText(img, "id{}".format(list(id2bbox.keys())[idx]), (int((x1 + x2)/2), int(y1)),
                        cv2.FONT_HERSHEY_PLAIN, 2, self.boxid_color, 2)
            if self.with_bbox:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.box_color, 2)
        return img

    def plot_skeleton_id(self, id2ske, img):
        for idx in range(len(id2ske)):
            # [x, y] = torch.mean(list(id2ske.values())[idx], dim=0)
            ske = list(id2ske.values())[idx]
            x = int(np.mean([ske[i] for i in range(len(ske)) if i % 2 == 0]))
            y = int(np.mean([ske[i] for i in range(len(ske)) if i % 2 != 0]))
            cv2.putText(img, "id{}".format(list(id2ske.keys())[idx]), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, self.skeid_color,
                        2)
        return img
