from models.TCN.test_TCN import TCNPredictor
from models.LSTM.test_LSTM import LSTMPredictor
import cv2
from collections import defaultdict
from models.ConvLSTM.test_ConvLstm import ConvLSTMPredictor
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
store_size = (540, 360)


class Tester:
    def __init__(self, model_name, video_path, label_path):
        self.tester = self.__get_tester(model_name)
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.kps_dict = defaultdict(list)
        self.coord = []
        self.pred = defaultdict(str)

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

    def test(self):
        cnt = 0
        while True:
            cnt += 1
            # print("Current frame is {}".format(cnt))
            ret, frame = self.cap.read()
            if ret:
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
                cv2.destroyAllWindows()
                break


class AutoTester:
    def __init__(self, models, videos, labels, res):
        self.models = [os.path.join(models, m) for m in os.listdir(models)]
        self.videos = [os.path.join(videos, v) for v in os.listdir(videos)]
        self.labels = [os.path.join(labels, l) for l in os.listdir(labels)]
        self.result_folder = res

    def test(self):
        for model in self.models:

            for v, l in zip(self.videos, self.labels):
                Tester(model, v, l)




if __name__ == '__main__':
    video = "1_video/test/others/1_Trim.mp4"
    Tester(model_path, video).test()