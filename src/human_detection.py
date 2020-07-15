import torch
import cv2
import copy
import numpy as np
from config import config


from src.estimator.pose_estimator import PoseEstimator
from src.estimator.visualize import KeyPointVisualizer
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from src.utils.utils import process_kp
from src.utils.img import torch_to_im, gray3D
from src.detector.box_postprocess import crop_bbox
from config.config import yolo_weight, yolo_cfg, pose_weight, pose_cfg

tensor = torch.FloatTensor


class HumanDetection:
    def __init__(self, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=yolo_cfg, weight=yolo_weight)
        self.object_tracker = ObjectTracker()
        self.pose_estimator = PoseEstimator(pose_cfg=pose_cfg, pose_weight=pose_weight)
        self.BBV = BBoxVisualizer()
        self.KPV = KeyPointVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.img_black = np.array([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps = {}
        self.kps_score = {}
        self.show_img = show_img

    def init_sort(self):
        self.object_tracker.init_tracker()

    def clear_res(self):
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.kps = {}
        self.kps_score = {}

    def visualize(self):
        img_black = cv2.imread('video/black.jpg')
        if config.plot_bbox and self.boxes is not None:
            self.frame = self.BBV.visualize(self.boxes, self.frame, self.boxes_scores)
            # cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))
        if config.plot_kps and self.kps is not []:
            self.frame = self.KPV.vis_ske(self.frame, self.kps, self.kps_score)
            img_black = self.KPV.vis_ske_black(self.frame, self.kps, self.kps_score)
        if config.plot_id and self.id2bbox is not None:
            self.frame = self.IDV.plot_bbox_id(self.id2bbox, self.frame)
            # frame = self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(img))
        return self.frame, img_black

    def process_img(self, frame, gray=False):
        self.clear_res()
        self.frame = frame

        with torch.no_grad():
            if gray:
                gray_img = gray3D(copy.deepcopy(frame))
                box_res = self.object_detector.process(gray_img)
            else:
                box_res = self.object_detector.process(frame)
            self.boxes, self.boxes_scores = self.object_detector.cut_box_score(box_res)

            if box_res is not None:
                self.id2bbox = self.object_tracker.track(box_res)
                boxes = self.object_tracker.id_and_box(self.id2bbox)

                inps, pt1, pt2 = crop_bbox(frame, boxes)
                kps, kps_score, kps_id = self.pose_estimator.process_img(inps, boxes, pt1, pt2)
                self.kps, self.kps_score = self.object_tracker.match_kps(kps_id, kps, kps_score)

        return self.kps, self.id2bbox, self.kps_score