import numpy as np
import math
import cv2
import torch
import pandas as pd
from collections import defaultdict
from config import config

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


def select_kps(idx, kps):
    if idx in kps.keys():
        return kps[idx]
    else:
        return [[]]


def dim2to1(raw_kp):
    kp = []
    for pt in raw_kp:
        for dim in pt:
            kp.append(dim)
    return kp


def reverse_csv(path):
    df = pd.read_csv(path, "r")
    data = df.values  # data是数组，直接从文件读出来的数据格式是数组
    index1 = list(df.keys())  # 获取原有csv文件的标题，并形成列表
    data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列表
    data = pd.DataFrame(data, index=index1)  # 将data的行列转换
    data.to_csv(path, header=0)


def numpy2str(array):
    s = ''
    for idx in range(len(array)):
        s += str(array[idx])+'\t'
    return s


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



def box2str(box):
    sub_box = ""
    for coor in box:
        sub_box += str(coor)
        sub_box += ","
    return sub_box[:-1]


def kp2str(kp):
    sub_kp = ""
    for item in kp:
        sub_kp += str(item[0])
        sub_kp += ","
        sub_kp += str(item[1])
        sub_kp += ","
    return sub_kp[:-1]


def kpScore2str(scores):
    scores = scores.tolist()
    sub_s = ""
    for item in scores:
        sub_s += str(item[0])
        sub_s += ","
    return sub_s[:-1]


def str2boxdict(s):
    d = defaultdict(list)
    id_bboxs = s.split("\t")
    for item in id_bboxs[:-1]:
        [idx, box] = item.split(":")
        bbox = box.split(",")
        d[idx] = [int(float(b)) for b in bbox]
    return d


def str2kpsdict(s):
    d = defaultdict(list)
    id_kps = s.split("\t")
    for item in id_kps[:-1]:
        [idx, rawkps] = item.split(":")
        kps_ls, kps = rawkps.split(","), []
        for i in range(config.body_part_num):
            kps.append([float(kps_ls[i*2]), float(kps_ls[i*2+1])])
        d[idx] = kps
    return d


def boxdict2str(k, v):
    boxstr = box2str(v)
    return "{}:{}\t".format(str(k), boxstr)


def kpsdict2str(k,v):
    kpstr = kp2str(v)
    return "{}:{}\t".format(str(k), kpstr)


def str2box(string):
    if string == "":
        return None
    tmp = string.split(",")
    boxes = []
    for item in tmp:
        boxes.append([float(i) for i in item.split(" ")])
    return boxes


def str2kps(string):
    if string == "":
        return None
    tmp = string.split(",")
    boxes = []
    for item in tmp:
        boxes.append([float(i) for i in item.split(" ")])
    return boxes


def kpsScoredict2str(k,v):
    kpstr = kpScore2str(v)
    return "{}:{}\t".format(str(k), kpstr)


def str2kpsScoredict(s):
    d = defaultdict()
    id_kps = s.split("\t")
    for item in id_kps[:-1]:
        [idx, raw_score] = item.split(":")
        tmp = [[float(item)] for item in raw_score.split(",")]
        score = torch.FloatTensor(tmp)
        d[idx] = score
    return d


if __name__ == '__main__':
    array = np.array([1,1,1,1,1])
    string = numpy2str(array)
    print(string)
    # ut = Utils()
    # # res = ut.time_to_string("10.0000")
    # # print(res)
    # res = ut.get_angle([0, 0], [1, -1], [0, 1])
    # print(res)
