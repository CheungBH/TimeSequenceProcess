from src.TCN.test_TCN import TCNPredictor
from src.LSTM.test_LSTM import LSTMPredictor
import cv2
from collections import defaultdict
from src.ConvLSTM.test_ConvLstm import ConvLSTMPredictor
from src.human_detection import ImgprocessorAllKPS as ImgProcessor
import numpy as np
from config import config
import os


model_folder = config.test_model_folder
video_folder = config.test_video_folder
result_folder = config.test_res_folder
label_folder = config.test_label_folder

cls = ["swim", "drown"]
seq_length = config.testing_frame
IP = ImgProcessor()
store_size = config.size


class Tester:
    def __init__(self, model_name, video_path, label_path):
        self.tester = self.__get_tester(model_name)
        self.video_name = video_path.split("/")[-1]
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.kps_dict = defaultdict(list)
        self.label, self.test_id = self.get_label(label_path)
        self.coord = []
        self.pred = defaultdict(str)
        self.pred_dict = defaultdict(list)
        self.res = defaultdict(list)

    def get_label(self, path):
        with open(path, "r") as lf:
            labels, ids = defaultdict(list), []
            for line in lf.readlines():
                [idx, label] = line[:-1].split(":")
                labels[idx] = [l for l in label.split(" ")]
                ids.append(idx)
        return labels, ids

    def __normalize_coordinates(self, coordinates):
        for i in range(len(coordinates)):
            if (i+1) % 2 == 0:
                coordinates[i] = coordinates[i] / self.height
            else:
                coordinates[i] = coordinates[i] / self.width
        return coordinates

    def __get_tester(self, model):
        if "ConvLSTM" in model:
            return ConvLSTMPredictor(model)
        if 'LSTM' in model:
            return LSTMPredictor(model)
        if "TCN" in model:
            return TCNPredictor(model)

    def __detect_kps(self):
        refresh_idx = []
        for k, v in self.kps_dict.items():
            if len(v) == seq_length:
                pred = self.tester.predict(np.array(v).astype(np.float32))
                self.pred[k] = cls[pred]
                self.pred_dict[str(k)].append(cls[pred])
                print("Predicting id {}".format(k))
                refresh_idx.append(k)
        for idx in refresh_idx:
            self.kps_dict[idx] = []

    def __put_pred(self, img):
        for idx, (k, v) in enumerate(self.pred.items()):
            cv2.putText(img, "id{}: {}".format(k,v), (30, int(40*(idx+1))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img

    def __put_cnt(self, img):
        for idx, (k, v) in enumerate(self.kps_dict.items()):
            cv2.putText(img, "id{} cnt: {}".format(k, len(v)), (300, int(30*(idx+2))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
        return img

    def __compare(self):
        assert self.pred_dict.keys() == self.label.keys()
        for k in self.pred_dict.keys():
            label, pred = self.label[k], self.pred_dict[k]
            assert len(label) == len(pred)
            for l, p in zip(label, pred):
                if p == "pass":
                    self.res[k].append("pass")
                self.res[k].append(l == p)
        return self.__summarize()

    def __summarize(self):
        res = {}
        for key, value in self.res.items():
            for idx, v in enumerate(value):
                sample_str = self.video_name[:-4] + "_id{}_frame{}-{}".format(key, 30*idx, 30*(idx+1)-1)
                res[sample_str] = v
        return res

    def test(self):
        cnt = 0
        while True:
            cnt += 1
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, store_size)
                kps, img = IP.process_img(frame)
                if kps:
                    for key in kps:
                        coord = self.__normalize_coordinates(kps[key])
                        self.kps_dict[key].append(coord)
                    self.__detect_kps()

                img = cv2.resize(img, store_size)
                img = self.__put_pred(img)
                cv2.putText(img, "Frame cnt: {}".format(cnt), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                img = self.__put_cnt(img)
                cv2.imshow("res", img)
                cv2.waitKey(2)

            else:
                self.cap.release()
                IP.init_sort()
                cv2.destroyAllWindows()
                break
        return self.__compare()


class AutoTester:
    def __init__(self, models, videos, labels):
        self.models = [os.path.join(models, m) for m in os.listdir(models)]
        self.videos = [os.path.join(videos, v) for v in os.listdir(videos)]
        self.labels = [os.path.join(labels, l) for l in os.listdir(labels)]
        self.final_res = defaultdict(list)

    def __merge_dict(self, res):
        for k, v in res:
            self.final_res[k] = v

    def test(self):
        for model in self.models:
            model_res = defaultdict()
            for v, l in zip(self.videos, self.labels):
                print("Begin processing {}".format(v))
                res = Tester(model, v, l).test()
                model_res.update(res)
            self.__merge_dict(model_res)
        return self.final_res


if __name__ == '__main__':
    # t = Tester("6_network/net_test/model/ConvLSTM_2020-03-09-17-28-39.pth", "tmp/v_1/video/50_Trim.mp4",
    #            "tmp/v_1/label1/50_Trim.txt")
    # rslt = t.test()
    # print(rslt)
    AT = AutoTester("tmp/test_v", "tmp/v2/video", "tmp/v2/label1")
    res = AT.test()
    print(res)


    # t.pred_dict["1"] = ["drown", "swim"]
    # t.pred_dict["2"] = ["drown"]
