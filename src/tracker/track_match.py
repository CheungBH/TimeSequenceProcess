from .sort import Sort
import torch
from utils.utils import Utils

Tensor = torch.cuda.FloatTensor


class ObjectTracker(object):
    def __init__(self):
        self.tracker = Sort()
        self.sorted = []
        self.track_boxes = []
        self.skeletons = []
        self.bboxes = []
        self.id2bbox = {}
        self.kp_score = {}
        self.id2ske = {}
        self.id2score = {}

    def init_tracker(self):
        self.tracker.init_KF()
    #
    # def __match(self):
    #     ske_center = {tuple(torch.mean(self.skeletons[idx], dim=0).tolist()): idx for idx in range(len(self.skeletons))}
    #     box_center = [((list(self.id2bbox.values())[idx][0] + list(self.id2bbox.values())[idx][2]) / 2,
    #                    (list(self.id2bbox.values())[idx][1] + list(self.id2bbox.values())[idx][3]) / 2)
    #                   for idx in range(len(self.id2bbox))]
    #     if len(ske_center) < len(box_center):
    #         id2ske = {}
    #         for idx, s_center in enumerate(ske_center.keys()):
    #             bs_dis = [Utils.cal_dis(s_center, b_center) for b_center in box_center]
    #             match_id = bs_dis.index(min(bs_dis))
    #             id2ske[list(self.id2bbox.keys())[match_id]] = self.skeletons[idx]
    #     else:
    #         sorted_skeleton = []
    #         for b_center in box_center:
    #             bs_dis = [Utils.cal_dis(s_center, b_center) for s_center in ske_center.keys()]
    #             match_id = bs_dis.index(min(bs_dis))
    #             sorted_skeleton.append(self.skeletons[match_id])
    #         id2ske = {int(list(self.id2bbox.keys())[idx]): sorted_skeleton[idx] for idx in range(len(self.id2bbox))}
    #
    #     self.__convert(id2ske)

    def __match(self):
        ske_center = [torch.mean(self.skeletons[idx], dim=0) for idx in range(len(self.skeletons))]
        box_center = [((list(self.id2bbox.values())[idx][0] + list(self.id2bbox.values())[idx][2]) / 2,
                       (list(self.id2bbox.values())[idx][1] + list(self.id2bbox.values())[idx][3]) / 2)
                      for idx in range(len(self.id2bbox))]
        if len(ske_center) < len(box_center):
            id2ske = {}
            for idx, s_center in enumerate(ske_center):
                bs_dis = [Utils.cal_dis(s_center, b_center) for b_center in box_center]
                match_id = bs_dis.index(min(bs_dis))
                id2ske[list(self.id2bbox.keys())[match_id]] = self.skeletons[idx]
        else:
            sorted_skeleton = []
            for b_center in box_center:
                bs_dis = [Utils.cal_dis(s_center, b_center) for s_center in ske_center]
                match_id = bs_dis.index(min(bs_dis))
                sorted_skeleton.append(self.skeletons[match_id])
            id2ske = {int(list(self.id2bbox.keys())[idx]): sorted_skeleton[idx] for idx in range(len(self.id2bbox))}

        self.match_kps_score(id2ske)
        self.__convert(id2ske)

    def __convert(self, id2ske):
        self.id2ske = {}
        for key, kps in id2ske.items():
            self.id2ske[key] = [[kps[i][0].tolist(), kps[i][1].tolist()] for i in range(len(kps))]

    def __track_bbox(self):
        box_tensor = Tensor([box + [0.999, 0.999, 0] for box in self.bboxes])
        self.tracked_bbox = self.tracker.update(box_tensor.cpu()).tolist()
        self.id2bbox = {int(box[4]): [box[0], box[1], box[2], box[3]] for box in self.tracked_bbox}

    def match_kps_score(self, id2ske):
        self.id2score = {}
        for i, (kp, score) in enumerate(zip(self.skeletons, self.kp_score)):
            for k, v in id2ske.items():
                if kp is v:
                    self.id2score[k] = score

    def track(self, bboxes, skeletons, kps_score):
        self.bboxes = bboxes.tolist()
        self.skeletons = skeletons
        self.kp_score = kps_score
        self.__track_bbox()
        self.__match()
        return self.id2ske, self.id2bbox, self.id2score

    def track_box(self, bboxes):
        box_tensor = Tensor([box + [0.999, 0.999, 0] for box in bboxes.tolist()])
        tracked_bbox = self.tracker.update(box_tensor.cpu()).tolist()
        id2bbox = {int(box[4]): [box[0], box[1], box[2], box[3]] for box in tracked_bbox}
        return id2bbox
