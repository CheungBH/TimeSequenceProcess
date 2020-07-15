import cv2
import torch
import numpy as np

from config.config import track_plot_id


class IDVisualizer(object):
    def __init__(self, with_bbox=True):
        self.with_bbox = with_bbox
        self.boxid_color = (255, 255, 0)
        self.box_color = (0, 0, 255)
        self.skeid_color = (0, 255, 255)

    def plot_bbox_id(self, id2bbox, img):
        for idx, box in id2bbox.items():
            if "all" not in track_plot_id:
                if idx not in track_plot_id:
                    continue
            [x1, y1, x2, y2] = box
            cv2.putText(img, "id{}".format(idx), (int((x1 + x2)/2), int(y1)), cv2.FONT_HERSHEY_PLAIN, 2, self.boxid_color, 2)
            if self.with_bbox:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.box_color, 2)
        return img

    def plot_skeleton_id(self, id2ske, img):
        for idx, kps in id2ske.items():
            if "all" not in track_plot_id:
                if idx not in track_plot_id:
                    continue
            # [x, y] = torch.mean(list(id2ske.values())[idx], dim=0)
            # x = int(np.mean([kps[i] for i in range(len(kps)) if i % 2 == 0]))
            # y = int(np.mean([kps[i] for i in range(len(kps)) if i % 2 != 0]))
            x = np.mean(np.array([item[0] for item in kps]))
            y = np.mean(np.array([item[1] for item in kps]))
            cv2.putText(img, "id{}".format(idx), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 2, self.skeid_color, 2)
        return img
