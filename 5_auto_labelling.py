import cv2
from src.human_detection import ImgProcessorNoPE as ImgProcessor
from collections import defaultdict
import os
from config import config

cls = config.label_cls
frame_length = config.label_frame
comment = config.label_comment

IP = ImgProcessor()
store_size = config.size


class LabelVideo:
    def __init__(self, video_path, id_ls, label_path):
        self.label_path = label_path
        self.id_ls = id_ls
        self.video_path = video_path
        self.idbox_cnt = defaultdict(int)
        self.label = defaultdict(list)
        self.cls = {str(idx): label for idx, label in enumerate(cls)}
        self.cls["p"] = "pass"

    def __put_cnt(self, img):
        for idx, (k, v) in enumerate(self.idbox_cnt.items()):
            cv2.putText(img, "id{} cnt: {}".format(k, v), (300, int(30*(idx+2))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 1)
        return img

    def __write_label(self):
        with open(self.label_path, "w") as lf:
            for k,v in self.label.items():
                label = " ".join([self.cls[name] for name in v])
                lf.write("{}:{}\n".format(k,label))

    def process(self):
        for num in self.id_ls:
            print("Begin processing id {}".format(num))
            cnt = 0
            self.idbox_cnt = defaultdict(int)
            cap = cv2.VideoCapture(self.video_path)
            while True:
                cnt += 1
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, store_size)
                    img, id2bbox = IP.process_img(frame)
                    img = cv2.resize(img, store_size)

                    if id2bbox:
                        for idx, (k, v) in enumerate(id2bbox.items()):
                            self.idbox_cnt[k] += 1
                        cv2.putText(img, "Frame cnt: {}".format(cnt), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                        self.__put_cnt(img)

                        if self.idbox_cnt[num] % frame_length == 0 and self.idbox_cnt[num] != 0:
                            self.label[num].append(input("The label of id {} is: ".format(num)))

                    cv2.imshow("res", img)
                    cv2.waitKey(2)

                else:
                    break

            cap.release()
            IP.init_sort()
            cv2.destroyAllWindows()

        self.__write_label()


class AutoLabel:
    def __init__(self, video_src, label_folder):
        self.video_ls = [os.path.join(video_src, "video", v_name) for v_name in os.listdir(os.path.join(video_src, "video"))]
        self.label_ls = [video_path.replace("video", label_folder)[:-4] +".txt" for video_path in self.video_ls]
        os.makedirs(video_src.replace("video", label_folder), exist_ok=True)
        self.ids = []
        self.label_log = os.path.join(video_src, "label_log.txt")

    def process(self):
        for v, l in zip(self.video_ls, self.label_ls):
            if os.path.exists(l):
                print("{} has been processed!".format(v))
                continue

            LV = LabelVideo(v, [1,2], l)
            print("Begin processing video {}".format(v))
            LV.process()
            print("Finish process\n")

        with open(self.label_log, "a+") as f:
            f.write(comment + "\n")


if __name__ == '__main__':
    video_s = "tmp/v2"
    label_name = "label1"
    os.makedirs(os.path.join(video_s, label_name), exist_ok=True)
    AL = AutoLabel(video_s, label_name)
    AL.process()
