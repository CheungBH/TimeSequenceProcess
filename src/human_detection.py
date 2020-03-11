from src.estimator.pose_estimator import PoseEstimator
from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
import torch
from src.tracker.sort import Sort
import cv2
import copy

Tensor = torch.cuda.FloatTensor


class ImgProcessor(object):
    def __init__(self, show_img=True):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetectionYolo()
        self.object_tracker = ObjectTracker()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.img_black = []
        self.show_img = show_img
        self.BBV = BBoxVisualizer()

    def init_sort(self):
        self.object_tracker.init_tracker()

    def __process_kp(self, kps):
        new_kp = []
        for bdp in range(len(kps)):
            for coord in range(2):
                new_kp.append(kps[bdp][coord])
        return new_kp

    def process_img(self, frame, get_id=1):
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(frame)
            if boxes is not None:
                key_points, img = self.pose_estimator.process_img(inps, orig_img, boxes, scores, pt1, pt2)
                img = self.BBV.visualize(boxes, img)
                if key_points:
                    id2ske, id2bbox = self.object_tracker.track(boxes, key_points)
                    img = self.IDV.plot_bbox_id(id2bbox, copy.deepcopy(img))
                    try:
                        kps = self.__choose_kps(id2ske, get_id)
                        # kps = self.__process_kp(id2ske[get_id])
                    except KeyError:
                        kps = []
                    return kps, img
                else:
                    return [], img
            else:
                return [], frame

    def __choose_kps(self, id2ske, get_id):
        return self.__process_kp(id2ske[get_id])


class ImgprocessorAllKPS:
    def __init__(self, show_img=True):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetectionYolo()
        self.object_tracker = ObjectTracker()
        self.IDV = IDVisualizer(with_bbox=False)
        self.img = []
        self.img_black = []
        self.show_img = show_img
        self.BBV = BBoxVisualizer()

    def init_sort(self):
        self.object_tracker.init_tracker()

    def __process_kp(self, kps):
        new_kp = []
        for bdp in range(len(kps)):
            for coord in range(2):
                new_kp.append(kps[bdp][coord])
        return new_kp

    def process_img(self, frame, get_id=1):
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(frame)
            if boxes is not None:
                key_points, img = self.pose_estimator.process_img(inps, orig_img, boxes, scores, pt1, pt2)
                img = self.BBV.visualize(boxes, img)
                if key_points:
                    id2ske, id2bbox = self.object_tracker.track(boxes, key_points)
                    img = self.IDV.plot_bbox_id(id2bbox, copy.deepcopy(img))
                    try:
                        kps = self.__choose_kps(id2ske, get_id)
                        # kps = self.__process_kp(id2ske[get_id])
                    except KeyError:
                        kps = []
                    return kps, img
                else:
                    return [], img
            else:
                return [], frame

    def __choose_kps(self, id2ske, get_id):
        res = {}
        for k, v in id2ske.items():
            res[k] = self.__process_kp(v)
        return res


class ImgProcessorNoPE(object):
    def __init__(self, show_img=True):
        self.object_detector = ObjectDetectionYolo()
        self.object_tracker = Sort()
        self.IDV = IDVisualizer(with_bbox=True)
        self.img = []
        self.img_black = []
        self.show_img = show_img
        self.BBV = BBoxVisualizer()

    def init_sort(self):
        self.object_tracker.init_KF()

    def __process_tracked_box(self, bbox):
        box_dict = {int(item[4]): item[:4] for item in bbox}
        return box_dict

    def process_img(self, frame, get_id=1):
        with torch.no_grad():
            inps, orig_img, boxes, scores, pt1, pt2 = self.object_detector.process(frame)
            if boxes is not None:
                box_tensor = Tensor([box.tolist() + [0.999, 0.999, 0] for box in boxes])
                tracked_bbox = self.object_tracker.update(box_tensor.cpu()).tolist()
                id2bbox = self.__process_tracked_box(tracked_bbox)
                # img = self.BBV.visualize(boxes, frame)
                img = self.IDV.plot_bbox_id(id2bbox, copy.deepcopy(frame))
                return img, id2bbox
            else:
                return frame, {}

#
# class ImgprocessorAllKPS(ImgProcessor):
#     def __init__(self):
#         super().__init__()
#         super().init_sort()
#         super().process_img(frame)
#
#
#     def __choose_kps(self, id2ske, get_id):
#         return id2ske

