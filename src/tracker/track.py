from .sort import Sort
import torch

tensor = torch.FloatTensor


class ObjectTracker(object):
    def __init__(self):
        self.tracker = Sort()

    def init_tracker(self):
        self.tracker.init_KF()

    def track_bbox(self, box_res):
        tracked_bbox = self.tracker.update(box_res.cpu())
        id2box = {int(box[4]): tensor(box[:4]) for box in tracked_bbox}
        return id2box

    def track_box_with_high_conf(self, box_res):
        bboxes = box_res.tolist()
        box_tensor = tensor([box + [0.999, 0.999, 0] for box in bboxes])
        tracked_bbox = self.tracker.update(box_tensor.cpu())
        id2box = {int(box[4]): tensor(box[:4]) for box in tracked_bbox}
        return id2box
