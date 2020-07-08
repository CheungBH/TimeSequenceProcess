from ..utils.img import cut_image
import cv2
import numpy as np
import torch
from .box_postprocess import cropBox, im_to_torch
try:
    from config.config import water_top
except:
    from src.debug.config.cfg_only_detections import water_top


class ImageProcessDetection:
    def __init__(self):
        self.water_top = water_top
        self.rect = []
        self.frame = []
        self.enhanced = []

    def __detect_people(self, diff):
        cut_diff = cut_image(diff, top=self.water_top)
        blur = cv2.blur(cut_diff, (7, 7))
        enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
        imageEnhance = cv2.filter2D(blur, -1, enhance_kernel)
        self.enhanced = imageEnhance
        hsv = cv2.cvtColor(imageEnhance, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 46])
        upper = np.array([180, 255, 255])
        thresh = cv2.inRange(hsv, lowerb=lower, upperb=upper)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilation = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, dilate_kernel)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stored = [idx for idx in range(len(contours)) if len(contours[idx]) > 80]
        real_con = [contours[i] for i in stored]
        self.rect = []
        for c in real_con:
            x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(self.frame, (x, y + self.water_top), (x + w, y + self.water_top + h), (0, 255, 0), 2)
            self.rect.append([x, y + self.water_top, x + w, y + self.water_top + h, 0.999, 0.999, 0])

    def detect_rect(self, diff):
        self.__detect_people(diff)
        return self.rect

    @staticmethod
    def __crop_from_dets(img, boxes, inps, pt1, pt2):
        imght = img.size(1)
        imgwidth = img.size(2)
        tmp_img = img
        tmp_img[0].add_(-0.406)
        tmp_img[1].add_(-0.457)
        tmp_img[2].add_(-0.480)
        for i, box in enumerate(boxes):
            upLeft = torch.Tensor((float(box[0]), float(box[1])))
            bottomRight = torch.Tensor((float(box[2]), float(box[3])))

            ht = bottomRight[1] - upLeft[1]
            width = bottomRight[0] - upLeft[0]

            scaleRate = 0.3
            upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
            upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
            bottomRight[0] = max(
                min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
            bottomRight[1] = max(
                min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

            try:
                inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, config.input_height, config.input_width)
            except IndexError:
                print(tmp_img.shape)
                print(upLeft)
                print(bottomRight)
                print('===')
            pt1[i] = upLeft
            pt2[i] = bottomRight
        return inps, pt1, pt2

    def process(self, frame):
        self.__detect_people(frame)
        bbox = torch.FloatTensor(self.rect)
        scores = torch.FloatTensor([0.99 for _ in range(len(bbox))])
        inps = torch.zeros(bbox.size(0), 3, 320, 256)
        pt1 = torch.zeros(bbox.size(0), 2)
        pt2 = torch.zeros(bbox.size(0), 2)
        inp = im_to_torch(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        inps, pt1, pt2 = self.__crop_from_dets(inp, bbox, inps, pt1, pt2)
        return inps, self.frame, self.rect, scores, pt1, pt2

