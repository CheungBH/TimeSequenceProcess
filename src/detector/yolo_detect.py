import torch
from config import config
from src.yolo.preprocess import prep_frame
from src.yolo.util import dynamic_write_results
from src.yolo.darknet import Darknet
from config.config import device
from ..utils.model_info import get_inference_time, print_model_param_nums, print_model_param_flops


class ObjectDetectionYolo(object):
    def __init__(self, cfg, weight, batchSize=1):
        self.det_model = Darknet(cfg)
        # self.det_model.load_state_dict(torch.load('models/yolo/yolov3-spp.weights', map_location="cuda:0")['model'])
        self.det_model.load_weights(weight)
        self.det_model.net_info['height'] = config.input_size
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        if device != "cpu":
            self.det_model.cuda()
        inf_time = get_inference_time(self.det_model, height=config.input_size, width=config.input_size)
        flops = print_model_param_flops(self.det_model, input_width=config.input_size, input_height=config.input_size)
        params = print_model_param_nums(self.det_model)
        print("Detection: Inference time {}s, Params {}, FLOPs {}".format(inf_time, params, flops))
        self.det_model.eval()

        self.im_dim_list = []
        self.batchSize = batchSize

    def __preprocess(self, frame):
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
        return img, im_dim_list

    def __detect(self, img, im_dim_list):
        self.im_dim_list = im_dim_list
        with torch.no_grad():
            # Human Detection
            if device != "cpu":
                img = img.cuda()

            prediction = self.det_model(img)
            # NMS process
            dets = dynamic_write_results(prediction, config.confidence,  config.num_classes, nms=True, nms_conf=config.nms_thresh)

            if isinstance(dets, int) or dets.shape[0] == 0:
                return None

            dets = dets.cpu()
            self.im_dim_list = torch.index_select(self.im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / self.im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * self.im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * self.im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
        return dets[:,1:]

    def process(self, frame):
        img, im_dim_list = self.__preprocess(frame)
        det_res = self.__detect(img, im_dim_list)
        # boxes, scores = self.cut_box_score(det_res)
        # return boxes, scores
        return det_res

    def cut_box_score(self, results):
        if results is None:
            return None, None

        for j in range(results.shape[0]):
            results[j, [0, 2]] = torch.clamp(results[j, [0, 2]], 0.0, self.im_dim_list[j, 0])
            results[j, [1, 3]] = torch.clamp(results[j, [1, 3]], 0.0, self.im_dim_list[j, 1])
        boxes = results[:, 0:4]
        scores = results[:, 4:5]

        # boxes_k = boxes[results[:, 0] == 0]
        # if isinstance(boxes, int) or boxes.shape[0] == 0:
        #     return None, None

        return boxes, scores

