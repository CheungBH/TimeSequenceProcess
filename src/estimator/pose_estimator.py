from config import config
from src.estimator.visualize import KeyPointVisualizer
from src.estimator.nms import pose_nms
from src.SPPE.src.main_fast_inference import *
from src.estimator.datatset import Mscoco
import cv2


class PoseEstimator(object):
    def __init__(self):
        self.skeleton = []
        self.KPV = KeyPointVisualizer()
        pose_dataset = Mscoco()
        if config.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()
        self.batch_size = config.pose_batch

    def __get_skeleton(self, boxes, scores, hm_data, pt1, pt2, orig_img):
        orig_img = np.array(orig_img, dtype=np.uint8)
        if boxes is None:
            return orig_img, [], [], boxes
        else:
            preds_hm, preds_img, preds_scores = getPrediction(
                hm_data, pt1, pt2, config.input_height, config.input_width, config.output_height, config.output_width)
            kps, score = pose_nms(boxes, scores, preds_img, preds_scores)

            if kps:
                img = self.KPV.vis_ske(orig_img, kps, score)
                img_black = self.KPV.vis_ske_black(orig_img, kps, score)
                return img, kps, img_black
            else:
                return orig_img, [], cv2.imread("src/black.jpg")

    def process_img(self, inps, orig_img, boxes, scores, pt1, pt2):
        try:
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % self.batch_size:
                leftover = 1
            num_batches = datalen // self.batch_size + leftover
            hm = []

            for j in range(num_batches):
                inps_j = inps[j * self.batch_size:min((j + 1) * self.batch_size, datalen)].cuda()
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm).cpu().data
            ske_img, skeleton, black_img = self.__get_skeleton(boxes, scores, hm, pt1, pt2, orig_img)
            return skeleton, ske_img, black_img
        except:
            return [], orig_img, cv2.imread("src/black.jpg")
