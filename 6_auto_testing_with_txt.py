from src.TCN.test_TCN import TCNPredictor
try:
    from src.LSTM.test_LSTM import LSTMPredictor
    lstm = True
except:
    lstm = False
from src.ConvGRU.test_ConvGRU import ConvGRUPredictor
from src.BiLSTM.test_BiLSTM import BiLSTMPredictor
import cv2
from src.tracker.visualize import IDVisualizer
from src.estimator.visualize import KeyPointVisualizer
from collections import defaultdict
from src.ConvLSTM.test_ConvLstm import ConvLSTMPredictor
from src.human_detection import HumanDetection as ImgProcessor
import numpy as np
from config import config
import os
from utils.kp_process import KPSProcessor
from utils.utils import str2boxdict, str2kpsdict, str2kpsScoredict
import csv

model_folder = config.test_model_folder
video_folder = config.test_video_folder
result_file = config.test_res_file
label_folder = config.test_label_folder
test_log = config.testing_log
write_video = config.test_write_video


with open(os.path.join("/".join(model_folder.split("/")[:-1]), "cls.txt"), "r") as cls_file:
    cls = []
    for line in cls_file.readlines():
        if "\n" in line:
            cls.append(line[:-1])
        else:
            cls.append(line)

seq_length = config.testing_frame
# IP = ImgProcessor()
store_size = config.size


class Tester:
    def __init__(self, model_name, video_path, label_path):
        model_name, video_path, label_path = model_name.replace("\\", "/"), video_path.replace("\\", "/"), label_path.replace("\\", "/")
        self.tester = self.__get_tester(model_name)
        self.video_name = video_path.split("/")[-1]
        self.cap = cv2.VideoCapture(video_path)
        self.IDV = IDVisualizer(with_bbox=True)
        self.KPV = KeyPointVisualizer()
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.kps_dict = defaultdict(list)
        self.KPSP = KPSProcessor(self.height, self.width)
        self.label, self.test_id = self.__get_label(label_path)
        self.coord = []
        self.pred = defaultdict(str)
        self.pred_dict = defaultdict(list)
        self.res = defaultdict(bool)
        self.label_dict = defaultdict(bool)
        with open("/".join(video_path.split("/")[:-1]) + "_txt/" + video_path.split("/")[-1][:-4] + "_box.txt", "r") as bf:
            self.box_txt = [line[:-1] for line in bf.readlines()]
        with open("/".join(video_path.split("/")[:-1]) + "_txt/" + video_path.split("/")[-1][:-4] + "_kps.txt", "r") as kf:
            self.kps_txt = [line[:-1] for line in kf.readlines()]
        with open("/".join(video_path.split("/")[:-1]) + "_txt/" + video_path.split("/")[-1][:-4] +
                  "_kps_score.txt", "r") as ksf:
            self.kps_score_txt = [line[:-1] for line in ksf.readlines()]
        if write_video:
            res_video = "/".join(video_path.split("/")[:-1]) + "_" + model_name.split("/")[-1][:-4] + "/" + self.video_name
            self.out = cv2.VideoWriter(res_video, cv2.VideoWriter_fourcc(*'XVID'), 10, store_size)

    def __get_label(self, path):
        with open(path, "r") as lf:
            labels, ids = defaultdict(list), []
            for line in lf.readlines():
                [idx, label] = line[:-1].split(":")
                labels[idx] = [l for l in label.split(" ")]
                ids.append(idx)
        return labels, ids

    def __get_tester(self, model):
        if "ConvLSTM" in model:
            return ConvLSTMPredictor(model, len(cls))
        if "BiLSTM" in model:
            return BiLSTMPredictor(model, len(cls))
        if "ConvGRU" in model:
            return ConvGRUPredictor(model, len(cls))
        if 'LSTM' in model:
            if lstm:
                return LSTMPredictor(model)
            else:
                print("lstm is not usable")
        if "TCN" in model:
            return TCNPredictor(model, len(cls))

    def __detect_kps(self):
        refresh_idx = []
        for k, v in self.kps_dict.items():
            if len(v) == seq_length:
                pred = self.tester.predict(np.array(v).astype(np.float32))
                self.pred[k] = cls[pred]
                self.pred_dict[str(k)].append(cls[pred])
                # print("Predicting id {}".format(k))
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

    def __get_target_pred(self):
        return {key: value for key, value in self.pred_dict.items() if key in self.test_id}

    def __compare(self):
        self.pred_dict = self.__get_target_pred()
        assert self.pred_dict.keys() == self.label.keys()
        for k in self.pred_dict.keys():
            label, pred = self.label[k], self.pred_dict[k]
            assert len(label) == len(pred)
            for idx, (l, p) in enumerate(zip(label, pred)):
                if l != "pass":
                    sample_str = self.video_name[:-4] + "_id{}_frame{}-{}".format(k, 30 * idx, 30 * (idx + 1) - 1)
                    self.res[sample_str] = l == p
                    self.label_dict[sample_str] = l

    def test(self):
        cnt = 0
        while True:
            cnt += 1
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, config.size)
                id2bbox = str2boxdict(self.box_txt.pop(0))
                id2kps = str2kpsdict(self.kps_txt.pop(0))
                id2kpsScore = str2kpsScoredict(self.kps_score_txt.pop(0))
                if id2bbox is not None:
                    frame = self.IDV.plot_bbox_id(id2bbox, frame)
                if id2kps is not None:
                    kps_tensor, score_tensor = self.KPV.kpsdic2tensor(id2kps, id2kpsScore)
                    frame = self.KPV.vis_ske(frame, kps_tensor, score_tensor)
                if id2kps:
                    for key, v in id2kps.items():
                        # coord = self.__normalize_coordinates(kps[key])
                        coord = self.KPSP.process_kp(v)
                        self.kps_dict[key].append(coord)
                    self.__detect_kps()

                img = frame
                img = cv2.resize(img, store_size)
                img = self.__put_pred(img)
                cv2.putText(img, "Frame cnt: {}".format(cnt), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                img = self.__put_cnt(img)
                cv2.imshow("res", img)
                cv2.waitKey(2)
                if write_video:
                    self.out.write(img)
            else:
                self.cap.release()
                # IP.init_sort()
                self.out.release()
                cv2.destroyAllWindows()
                break
        self.__compare()
        return self.res


class AutoTester:
    def __init__(self, models, videos, labels):
        self.video_path = videos
        self.models = [os.path.join(models, m) for m in os.listdir(models)]
        self.videos = sorted([os.path.join(videos, v) for v in os.listdir(videos)])
        self.labels = sorted([os.path.join(labels, l) for l in os.listdir(labels)])
        self.final_res = defaultdict(list)
        self.model_name = []
        self.label_dict = defaultdict(bool)

    def __merge_dict(self, res):
        for k, v in res.items():
            self.final_res[k].append(v)
        print(self.final_res)

    def test(self):
        model_cnt = 0
        write_gt = True
        for model in self.models:
            model = model.replace("\\", "/")
            model_cnt += 1
            model_res = defaultdict()
            if write_video:
                os.makedirs(self.video_path + "_" + model.split("/")[-1][:-4], exist_ok=True)
            print("\n[{}/{}] ---> Testing model {}".format(model_cnt, len(self.models), model))
            self.model_name.append(model.split("\\")[-1])
            video_cnt = 0
            for v, l in zip(self.videos, self.labels):
                video_cnt += 1
                print("--- [{}/{}]. processing video {}".format(video_cnt, len(self.videos), v))
                tester = Tester(model, v, l)
                res = tester.test()
                if write_gt:
                    self.label_dict.update(tester.label_dict)
                model_res.update(res)
            # print(model_res)
            write_gt = False
            self.__merge_dict(model_res)
        return self.final_res


def write_result(result, model_name, out, gt):
    f = open(out, "w", newline="")
    csv_writer = csv.writer(f)
    out = ["model_name", "ground truth"]
    for model in model_name:
        out.append(model)
    csv_writer.writerow(out)

    for k, v in result.items():
        out = [k, gt[k]]
        for res in v:
            out.append(res)
        csv_writer.writerow(out)
    f.close()


if __name__ == '__main__':
    print("\n\n\n\nChecking information...")
    print("Your model folder --------> {}".format(model_folder))
    print("Your video folder --------> {}".format(video_folder))
    print("Your label folder --------> {}".format(label_folder))
    print("Your result file name --------> {}".format(result_file))
    input("Press any keys to continue")

    # t = Tester("tmp/models/TCN_struct1_2020-03-09-18-03-19.pth", "tmp/v_1/video/50_Trim.mp4",
    #            "tmp/v_1/label1/50_Trim.txt")
    # rslt = t.test()
    # print(rslt)
    # AT = AutoTester("tmp/net1", "tmp/v_1/video", "tmp/v_1/label1")
    # test_result, model_name = AT.test()
    # write_result(test_result, model_name, "tmp/out_test.csv")
    os.makedirs("/".join(result_file.split('/')[:-1]), exist_ok=True)
    AT = AutoTester(model_folder, video_folder, label_folder)
    test_result = AT.test()
    model_name = AT.model_name
    ground_truth = AT.label_dict
    write_result(test_result, model_name, result_file, ground_truth)

    with open(os.path.join(result_file.split("/")[0], "description.txt"), "a+") as f:
        f.write(test_log)

    # t.pred_dict["1"] = ["drown", "swim"]
    # t.pred_dict["2"] = ["drown"]
