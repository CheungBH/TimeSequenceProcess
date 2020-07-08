import cv2


class BBoxVisualizer(object):
    def __init__(self):
        self.box_color = (0, 0, 255)
        self.score_color = (255, 255, 168)

    def visualize(self, bboxes, img, scores=None):
        if bboxes is None:
            return img

        if isinstance(bboxes, dict):
            bboxes = self.__dict2ls(bboxes)

        scores = self.__unify_type(scores)

        for idx, bbox in enumerate(bboxes):
            [x1, y1, x2, y2] = bbox
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), self.box_color, 4)
            if scores is not None:
                cv2.putText(img, "{}".format(round(scores[idx], 2)), (int((x1+x2)/2), int((y1+y2)/2)),
                            cv2.FONT_HERSHEY_PLAIN, 2, self.score_color, 2)
        return img

    def __dict2ls(self, id2box):
        return [v for k, v in id2box.items()]

    def __unify_type(self, scores):
        if scores is not None:
            if not isinstance(scores, list):
                if len(scores.shape) > 1:
                    scores = scores.squeeze(1)
                scores = scores.tolist()

        return scores