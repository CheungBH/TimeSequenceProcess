from .sort import Sort
import torch

tensor = torch.FloatTensor


class ObjectTracker(object):
    def __init__(self):
        self.tracker = Sort()
        self.ids = []
        self.id2box = {}
        self.boxes = []

    def init_tracker(self):
        self.tracker.init_KF()

    def clear(self):
        self.ids = []
        self.id2box = {}
        self.boxes = []

    def track(self, box_res):
        self.clear()
        tracked_box = self.tracker.update(box_res.cpu())
        self.id2box = {int(box[4]): tensor(box[:4]) for box in tracked_box}
        # self.id2box = sorted(tracked_box.items(),key=lambda x:x[0])
        return self.id2box

    def id_and_box(self, tracked_box):
        boxes = sorted(tracked_box.items(), key=lambda x: x[0])
        self.ids = [item[0] for item in boxes]
        self.boxes = [item[1].tolist() for item in boxes]
        return tensor(self.boxes)

    def match_kps(self, kps_id, kps, kps_score):
        id2kps, id2kpScore = {}, {}
        for idx, (kp_id) in enumerate(kps_id):
            id2kps[self.ids[kp_id]] = kps[idx]
            id2kpScore[self.ids[kp_id]] = kps_score[idx]
        return id2kps, id2kpScore

    def match(self, kps, kps_score):
        id2box, id2kps, id2kpScore = {}, {}, {}
        for item in self.tracked_box:
            mark1, mark2 = item[0].tolist(), item[1].tolist()
            for j in range(len(self.box)):
                if self.box[j][0].tolist() == mark1 and self.box[j][1].tolist() == mark2:
                    idx = item[4]
                    id2box[idx] = item[:4]
                    id2kps[idx] = kps[j]
                    id2kpScore = kps_score[j]
        return id2kps, id2box, id2kpScore
    #
    # def track_box(self):
    #     return {int(box[4]): tensor(box[:4]) for box in self.tracked_box}

