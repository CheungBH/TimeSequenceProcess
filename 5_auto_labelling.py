
import cv2
from src.human_detection import HumanDetection
import os
from utils.utils import *
from config import config

cls = config.label_cls
frame_length = config.label_frame
comment = config.label_comment
videos = config.label_main_folder
labels = config.label_folder_name

IP = HumanDetection()
store_size = config.size
write = config.write_label_info


class LabelVideo:
    def __init__(self, video_path, label_path):
        self.label_path = label_path.replace("\\", "/")
        self.video_path = video_path.replace("\\", "/")
        self.idbox_cnt = defaultdict(int)
        self.label = defaultdict(list)
        self.cls = {str(idx): label for idx, label in enumerate(cls)}
        self.cls["p"] = "pass"
        self.cls_str = ""
        for k, v in self.cls.items():
            self.cls_str += "{}-->{}, ".format(k,v)
        self.id_record = defaultdict(bool)
        if write:
            txt_folder = "/".join(self.video_path.split("/")[:-1]) + "_txt"
            video_name = self.video_path.split("/")[-1][:-4]
            os.makedirs(txt_folder, exist_ok=True)
            self.box_txt = open(os.path.join(txt_folder, "{}_box.txt".format(video_name)), "w")
            self.kps_score_txt = open(os.path.join(txt_folder, "{}_kps_score.txt".format(video_name)), "w")
            self.kps_txt = open(os.path.join(txt_folder, "{}_kps.txt".format(video_name)), "w")

    def __put_cnt(self, img):
        for idx, (k, v) in enumerate(self.idbox_cnt.items()):
            cv2.putText(img, "id{} cnt: {}".format(k, v), (300, int(30*(idx+2))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 1)
        return img

    def __write_label(self):
        with open(self.label_path, "a+") as lf:
            for k,v in self.label.items():
                label = " ".join([self.cls[name] for name in v])
                lf.write("{}:{}\n".format(k,label))

    def assign_label(self, num):
        return input("({}) The label of id {} is: ".format(self.cls_str[:-2], num))

    def if_continue(self):
        return input("Continue labelling other ids? ('no' to break)")

    def next_id(self):
        return input("Please input the target id:")

    def __write_box(self, boxes):
        if boxes is not None:
            if write:
                box_str = ""
                for k, v in boxes.items():
                    box_str += boxdict2str(k, v)
                self.box_txt.write(box_str)
                self.box_txt.write("\n")
        else:
            if write:
                self.box_txt.write("\n")

    def __write_kps(self, kps, score):
        if kps:
            if write:
                kps_str = ""
                for k, v in kps.items():
                    kps_str += kpsdict2str(k, v)
                self.kps_txt.write(kps_str)
                self.kps_txt.write("\n")

                kps_score_str = ""
                for k, v in score.items():
                    kps_score_str += kpsScoredict2str(k, v)
                self.kps_score_txt.write(kps_score_str)
                self.kps_score_txt.write("\n")
        else:
            if write:
                self.kps_txt.write("\n")
                self.kps_score_txt.write("\n")

    def process(self):
        num = 1
        while True:
            print("Begin processing id {}".format(num))
            cnt = 0
            self.idbox_cnt = defaultdict(int)
            cap = cv2.VideoCapture(self.video_path)
            recorded = True
            while True:
                cnt += 1
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, store_size)
                    kps, box, kpScore = IP.process_img(frame)
                    img, _ = IP.visualize()

                    self.__write_box(box)
                    self.__write_kps(kps, kpScore)

                    cv2.putText(img, "Frame cnt: {}".format(cnt), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 1)
                    if kps:
                        for idx, (k, v) in enumerate(kps.items()):
                            self.idbox_cnt[k] += 1
                        self.__put_cnt(img)

                        if num in kps.keys():
                            if self.idbox_cnt[num] % frame_length == 0 and self.idbox_cnt[num] != 0 and recorded == False:
                                inp = self.assign_label(num)
                                recorded = True
                                while inp not in self.cls:
                                    inp = input("Your input is not right! The label of id {} is: ".format(num))
                                self.label[num].append(inp)
                            else:
                                if self.idbox_cnt[num] % frame_length != 0:
                                    recorded = False

                    cv2.imshow("res", img)
                    cv2.waitKey(2)

                else:
                    exist_ids = self.idbox_cnt.keys()
                    break

            cap.release()
            IP.init_sort()
            cv2.destroyAllWindows()

            if_continue = self.if_continue()
            if if_continue == "no":
                break
            else:
                while True:
                    num = self.next_id()
                    try:
                        num = eval(num)
                        if num in exist_ids:
                            break
                        else:
                            print("Your id assigned is not included in the exist ids!")
                    except:
                        print("Please input number!")
                        continue
                continue

        self.__write_label()


class AutoLabel:
    def __init__(self, video_src, label_folder):
        self.video_ls = [os.path.join(video_src, "video", v_name) for v_name in os.listdir(os.path.join(video_src, "video"))]
        self.label_ls = [video_path.replace("video", label_folder)[:-4] +".txt" for video_path in self.video_ls]
        os.makedirs(video_src.replace("video", label_folder), exist_ok=True)
        self.label_log = os.path.join(video_src, "label_log.txt")

    def process(self):
        video_cnt = 0
        for v, l in zip(self.video_ls, self.label_ls):
            video_cnt += 1
            if os.path.exists(l):
                print("--- [{}/{}] {} has been processed!\n".format(video_cnt, len(self.video_ls),v))
                continue

            LV = self.get_labeler(v, l)
            print("--- [{}/{}] Begin processing video {}".format(video_cnt, len(self.video_ls),v))
            LV.process()
            print("Finish\n")

        with open(self.label_log, "a+") as f:
            f.write(comment + "\n")

    def get_labeler(self, video, label):
        return LabelVideo(video, label)


class LabelVideoWithSameLabel(LabelVideo):
    def __init__(self, label, ids, video_src, label_folder):
        super().__init__(video_src, label_folder)
        self.label_str = label
        self.cls_num = {c: str(i) for i, c in enumerate(cls)}
        self.ids = ids
        self.cnt = 0

    def assign_label(self, num):
        print("The action of id {} is {}".format(num, self.label_str))
        return self.cls_num[self.label_str]

    def next_id(self):
        return self.ids[self.cnt]

    def if_continue(self):
        self.cnt += 1
        if self.cnt >= len(self.ids):
            return "no"
        return "yes"


class AutoLabelWithSameLabel(AutoLabel):
    def __init__(self, video_src, label_folder, label, ids=(1,)):
        super().__init__(video_src, label_folder)
        self.all_label = label
        self.ids = ids

    def get_labeler(self, video, label):
        return LabelVideoWithSameLabel(self.all_label, self.ids, video, label)


if __name__ == '__main__':
    # videos = videos
    # labels = "label4"
    os.makedirs(os.path.join(videos, labels), exist_ok=True)
    AL = AutoLabel(videos, labels)
    # AL = AutoLabelWithSameLabel(videos, labels, "swim")
    AL.process()
