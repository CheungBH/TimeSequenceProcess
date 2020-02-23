from src.estimator.pose_estimator import PoseEstimator
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
import torch
import cv2
import copy


class ImgProcessor:
    def __init__(self, show_img=True):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetectionYolo()
        self.img = []
        self.img_black = []
        self.show_img = show_img
        self.BBV = BBoxVisualizer()

    def __process_kp(self, kps):
        kps = kps[0].tolist()
        new_kp = []
        for bdp in range(len(kps)):
            for coord in range(2):
                new_kp.append(kps[bdp][coord])
        return new_kp

    def process_img(self, frame):
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(frame)
            if boxes is not None:
                key_points, img = self.pose_estimator.process_img(inps, orig_img, boxes, scores, pt1, pt2)
                cv2.imshow("bbox", self.BBV.visualize(boxes, copy.deepcopy(frame)))
                if key_points:
                    kps = self.__process_kp(key_points)
                    return kps, img
                else:
                    return [], img
            else:
                return [], frame

