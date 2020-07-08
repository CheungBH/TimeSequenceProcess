from .visualize import KeyPointVisualizer
from .nms import pose_nms
from .datatset import Mscoco
from config import config
from ..utils.model_info import *
import torch
from ..utils.eval import getPrediction


class PoseEstimator(object):
    def __init__(self, pose_cfg, pose_weight):
        self.KPV = KeyPointVisualizer()
        pose_dataset = Mscoco()

        if config.pose_backbone == "seresnet101":
            from src.pose_model.seresnet.FastPose import InferenNet_fast as createModel
            self.pose_model = createModel(4 * 1 + 1, pose_dataset, pose_weight, cfg=pose_cfg)
        elif config.pose_backbone == "mobilenet":
            from src.pose_model.mobilenet.MobilePose import createModel
            self.pose_model = createModel(cfg=config.pose_cfg)
        else:
            raise ValueError("Not a backbone!")
        if config.device != "cpu":
            self.pose_model.cuda()
            self.pose_model.eval()
        inf_time = get_inference_time(self.pose_model, height=config.input_height, width=config.input_width)
        flops = print_model_param_flops(self.pose_model)
        params = print_model_param_nums(self.pose_model)
        print("Pose estimation: Inference time {}s, Params {}, FLOPs {}".format(inf_time, params, flops))
        self.batch_size = config.pose_batch

    def process_img(self, inps, boxes, scores, pt1, pt2):
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % self.batch_size:
            leftover = 1
        num_batches = datalen // self.batch_size + leftover
        hm = []

        for j in range(num_batches):
            if config.device != "cpu":
                inps_j = inps[j * self.batch_size:min((j + 1) * self.batch_size, datalen)].cuda()
            else:
                inps_j = inps[j * self.batch_size:min((j + 1) * self.batch_size, datalen)]
            hm_j = self.pose_model(inps_j)
            hm.append(hm_j)
        hm = torch.cat(hm).cpu().data

        preds_hm, preds_img, preds_scores = getPrediction(
            hm, pt1, pt2, config.input_height, config.input_width, config.output_height, config.output_width)
        kps, kps_score = pose_nms(boxes, scores, preds_img, preds_scores)

        return kps, kps_score

