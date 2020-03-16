import numpy as np
import math
import cv2
import torch
import pandas as pd


image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


def reverse_csv(path):
    df = pd.read_csv(path, "r")
    data = df.values  # data是数组，直接从文件读出来的数据格式是数组
    index1 = list(df.keys())  # 获取原有csv文件的标题，并形成列表
    data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列表
    data = pd.DataFrame(data, index=index1)  # 将data的行列转换
    data.to_csv(path, header=0)


def __separate_sample(sample):
    data, label = [], []
    for item in sample:
        data.append(item[0])
        label.append(item[1])
    return data, label


def __ls_preprocess(ls):
    try:ls.remove("\n")
    except: pass
    while True:
        try: ls.remove("")
        except ValueError: break
    return ls


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
    file = "../tmp/train_video_res.csv"
    reverse_csv(file)
    # ut = Utils()
    # # res = ut.time_to_string("10.0000")
    # # print(res)
    # res = ut.get_angle([0, 0], [1, -1], [0, 1])
    # print(res)
