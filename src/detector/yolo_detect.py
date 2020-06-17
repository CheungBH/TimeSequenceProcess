import torch
from config import config
from src.yolo.preprocess import prep_frame
from src.yolo.util import dynamic_write_results
from src.yolo.darknet import Darknet
from config.config import device


class ObjectDetectionYolo(object):
    def __init__(self, batchSize=1):
        self.det_model = Darknet(config.yolo_cfg)
        # self.det_model.load_state_dict(torch.load('models/yolo/yolov3-spp.weights', map_location="cuda:0")['model'])
        self.det_model.load_weights(config.yolo_weights)
        self.det_model.net_info['height'] = config.input_size
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        if device != "cpu":
            self.det_model.cuda()
        self.det_model.eval()

        self.stopped = False
        self.batchSize = batchSize

    def __video_process(self, frame):
        img = []
        orig_img = []
        # im_name = []
        im_dim_list = []
        img_k, orig_img_k, im_dim_list_k = prep_frame(frame, int(config.input_size))

        img.append(img_k)
        orig_img.append(orig_img_k)
        # im_name.append('0.jpg')
        im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            # Human Detection
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        return img, orig_img, im_dim_list

    def __get_bbox(self, img, orig_img, im_dim_list):
        with torch.no_grad():
            # Human Detection
            if device != "cpu":
                img = img.cuda()
                prediction = self.det_model(img, CUDA=True)
            else:
                prediction = self.det_model(img, CUDA=False)
            # NMS process
            dets = dynamic_write_results(prediction, config.confidence,  config.num_classes, nms=True, nms_conf=config.nms_thresh)

            if isinstance(dets, int) or dets.shape[0] == 0:
                return orig_img[0], None, None

            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return orig_img[0], None, None

        return orig_img[0], boxes_k, scores[dets[:, 0] == 0]

    def process(self, frame):
        img, orig_img, im_dim_list = self.__video_process(frame)
        orig_img, boxes, scores = self.__get_bbox(img, orig_img, im_dim_list)
        # inps, orig_img, boxes, scores, pt1, pt2 = crop_bbox(orig_img, boxes, scores)
        return orig_img, boxes, scores

