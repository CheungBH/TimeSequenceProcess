import cv2
from src.human_detection import ImgProcessorNoPE as ImgProcessor
from collections import defaultdict
import os
from config import config

cls = config.label_cls
IP = ImgProcessor()
store_size = (540, 360)
frame_length = config.label_frame


class LabelVideo:
    def __init__(self, video_path, id_ls, label_folder):
        self.label_path = (video_path.replace("video", label_folder))[:-4] + ".txt"
        self.id_ls = id_ls
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.idbox_cnt = defaultdict(int)
        self.label = defaultdict(list)

    def __put_cnt(self, img):
        for idx, (k, v) in enumerate(self.idbox_cnt.items()):
            cv2.putText(img, "id{} cnt: {}".format(k, v), (300, int(30*(idx+2))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 1)
        return img

    def __write_label(self):
        with open(self.label_path, "w") as lf:
            for k,v in self.label.items():
                label =
                lf.write("label")

    def process(self):
        cnt = 0
        for id in self.id_ls:
            while True:
                cnt += 1
                # print("Current frame is {}".format(cnt))
                ret, frame = self.cap.read()
                if ret:
                    img, id2bbox = IP.process_img(frame)
                    img = cv2.resize(img, store_size)

                    if id2bbox:
                        for idx, (k, v) in enumerate(id2bbox.items()):
                            self.idbox_cnt[k] += 1
                        cv2.putText(img, "Frame cnt: {}".format(cnt), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                        self.__put_cnt(img)

                        if self.idbox_cnt[id] % frame_length == 0:
                            self.label[id].append(input())

                    cv2.imshow("res", img)
                    cv2.waitKey(2)

                else:
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break

        self.__write_label()
        return self.label


if __name__ == '__main__':
    video = "7_test/test1/video/3c_Trim.mp4"
    l_folder = "label_1"
    os.makedirs("/".join(video.split("/")[:-2]) + "/" + l_folder)
    label = LabelVideo(video, [1, 3], "label_1").process()
    print(label)
