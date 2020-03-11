import numpy as np
import math
import cv2
import torch

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


class OneHotConverter:
    def __init__(self, maxlabel):
        self.max = maxlabel
        self.vector = [0] * self.max

    def convert(self, idx):
        self.vector[idx] = 1
        vector = self.vector
        self.vector = [0] * self.max
        return vector


class Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def get_angle(center_coor, coor2, coor3):
        L1 = Utils.cal_dis(coor2,coor3)
        L2 = Utils.cal_dis(center_coor,coor3)
        L3 = Utils.cal_dis(center_coor,coor2)
        Angle = Utils.cal_angle(L1,L2,L3)
        return Angle

    @staticmethod
    def cal_dis(coor1, coor2):
        out = np.square(coor1[0] - coor2[0]) + np.square(coor1[1] - coor2[1])
        return np.sqrt(out)

    @staticmethod
    def cal_angle(L1, L2, L3):
        out = (np.square(L2) + np.square(L3) - np.square(L1)) / (2 * L2 * L3)
        try:
            return math.acos(out) * (180 / math.pi)
        except ValueError:
            return 180


if __name__ == '__main__':
    pass
    # ut = Utils()
    # # res = ut.time_to_string("10.0000")
    # # print(res)
    # res = ut.get_angle([0, 0], [1, -1], [0, 1])
    # print(res)
