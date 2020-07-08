import torch
from ..utils.img import cropBox, im_to_torch
from config import config
import cv2


def crop_bbox(orig_img, boxes):
    with torch.no_grad():
        if orig_img is None:
            return None, None, None

        if boxes is None or boxes.nelement() == 0:
            return None, None, None

        inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        # inp = orig_img
        inps, pt1, pt2 = crop_from_dets(inp, boxes)
        return inps, pt1, pt2


def crop_from_dets(img, boxes):

    inps = torch.zeros(boxes.size(0), 3, config.input_height, config.input_width)
    pt1 = torch.zeros(boxes.size(0), 2)
    pt2 = torch.zeros(boxes.size(0), 2)

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


def merge_box(gray_box, black_box, gray_scores, black_scores):
    return gray_box, gray_scores

