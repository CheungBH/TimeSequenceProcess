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


if __name__ == '__main__':
    pass
    # ut = Utils()
    # # res = ut.time_to_string("10.0000")
    # # print(res)
    # res = ut.get_angle([0, 0], [1, -1], [0, 1])
    # print(res)
