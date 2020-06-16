from src.TCN.test_TCN import TCNPredictor
# from src.LSTM.test_LSTM import LSTMPredictor
from src.ConvGRU.test_ConvGRU import ConvGRUPredictor
from src.BiLSTM.test_BiLSTM import BiLSTMPredictor
import cv2
from collections import defaultdict
from src.ConvLSTM.test_ConvLstm import ConvLSTMPredictor
from src.human_detection import ImgprocessorAllKPS as ImgProcessor
import numpy as np
from config import config
import os
import csv

model_folder = config.test_model_folder
video_folder = config.test_video_folder
result_file = config.test_res_file
label_folder = config.test_label_folder
test_log = config.testing_log

with open(os.path.join("/".join(model_folder.split("/")[:-1]), "cls.txt"), "r") as cls_file:
    cls = []
    for line in cls_file.readlines():
        if "\n" in line:
            cls.append(line[:-1])
        else:
            cls.append(line)

seq_length = config.testing_frame
IP = ImgProcessor()
store_size = config.size


class Tester:
    def __init__(self, model_name, video_path, label_path):
        self.tester = self.__get_tester(model_name)
        self.video_name = video_path.split("\\")[-1]
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.kps_dict = defaultdict(list)
        self.label, self.test_id = self.__get_label(label_path)
        self.coord = []
        self.pred = defaultdict(str)
        self.pred_dict = defaultdict(list)
        self.res = defaultdict(bool)
        self.label_dict = defaultdict(bool)

    def __get_label(self, path):
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
            return ConvLSTMPredictor(model, len(cls))
        if "BiLSTM" in model:
            return BiLSTMPredictor(model, len(cls))
        if "ConvGRU" in model:
            return ConvGRUPredictor(model, len(cls))
        if 'LSTM' in model:
            return LSTMPredictor(model)
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
        self.__compare()
        return self.res


class AutoTester:
    def __init__(self, models, videos, labels):
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
            model_cnt += 1
            model_res = defaultdict()
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
