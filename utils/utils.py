import numpy as np
import math
import cv2
import torch

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


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

    @staticmethod
    def count_average(origin_ls):
        length = len(origin_ls)
        if length != 0:
            x_all = 0
            y_all = 0
            for i in range(int(length/2)):
                x_all += origin_ls[2*i]
                y_all += origin_ls[2*i +1]
            x_ave = x_all*2 / length
            y_ave = y_all*2 / length
            return [x_ave, y_ave]
        else:
            return []

    @staticmethod
    def image_normalize(image, size=224):
        image_array = cv2.resize(image, (size, size))
        image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
        image_array = image_array.transpose((2, 0, 1))
        for channel, _ in enumerate(image_array):
            image_array[channel] /= 255.0
            image_array[channel] -= image_normalize_mean[channel]
            image_array[channel] /= image_normalize_std[channel]
        image_tensor = torch.from_numpy(image_array).float()
        return image_tensor

    @staticmethod
    def get_decimal(string):
        if "." in string:
            idx = string.index(".")
            if len(string[idx+1:]) >= 2:
                return string[idx+1: idx+3]
            else:
                return string[idx+1:] + "0"*(2-len(string[idx+1:]))
        else:
            return "00"

    @staticmethod
    def get_integer(string):
        if "." in string:
            idx = string.index(".")
            if len(string[:idx]) >= 2:
                return string[idx-2: idx]
            else:
                return "0"*(2-len(string[:idx])) + string[:idx]
        else:
            return string.zfill(2)

    @staticmethod
    def to_unity_string(string):
        integer = Utils.get_integer(str(string))
        decimal = Utils.get_decimal(str(string))
        return integer + decimal

    @staticmethod
    def point_to_string(point):
        point = str(point * 100)
        return str(point[:4]).zfill(4)

    @staticmethod
    def time_to_string(time):
        integer = str(int(time)).zfill(2)
        decimal = Utils.get_decimal(time)
        return integer + str(decimal * 100)[:2].zfill(2)


if __name__ == '__main__':
    ut = Utils()
    # res = ut.time_to_string("10.0000")
    # print(res)
    res = ut.get_angle([0, 0], [1, -1], [0, 1])
    print(res)
