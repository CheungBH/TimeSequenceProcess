import cv2
import torch
import numpy as np


image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


def process_kp(kps, idx):
    new_kp = []
    for bdp in range(len(kps)):
        for coord in range(2):
            new_kp.append(kps[bdp][coord])
    return {idx: new_kp}


def image_normalize(img_name, size=224):
    if isinstance(img_name, str):
        image_array = cv2.imread(img_name)
    else:
        image_array = img_name
    image_array = cv2.resize(image_array, (size, size))
    image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
    image_array = image_array.transpose((2, 0, 1))
    for channel, _ in enumerate(image_array):
        image_array[channel] /= 255.0
        image_array[channel] -= image_normalize_mean[channel]
        image_array[channel] /= image_normalize_std[channel]
    image_tensor = torch.from_numpy(image_array).float()
    return image_tensor


def cal_center_point(box):
    return int((box[2] - box[0]) / 2) + box[0], int((box[3] - box[1]) / 2) + box[1]