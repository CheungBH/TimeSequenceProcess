from src.TCN.test_TCN import TCNPredictor
from src.LSTM.test_LSTM import LSTMPredictor
import cv2
from collections import defaultdict
from src.ConvLSTM.test_ConvLstm import ConvLSTMPredictor
from src.human_detection import ImgprocessorAllKPS as ImgProcessor
import numpy as np
import os


cls = ["drown_side","Stand","swim"]
num_classes = len(cls)
seq_length = 30
IP = ImgProcessor()
store_size = (540, 360)


class Tester:
    def __init__(self, model_name, video_path):
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
            return ConvLSTMPredictor(model, num_classes)
        if 'LSTM' in model:
            return LSTMPredictor(model)
        if "TCN" in model:
            return TCNPredictor(model, num_classes)

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

    def __detect_kps_pro(self):
        refresh_idx = []
        for k, v in self.kps_dict.items():
            if len(v) == seq_length:
                pred = self.tester.predict_pos(np.array(v).astype(np.float32))
                if pred == 0:
                    self.pred[k] = "drown"
                else:
                    self.pred[k] = "others"
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
                frame = cv2.resize(frame, store_size)
                kps, img,_,_,_ = IP.process_img(frame)
                if kps:
                    for key in kps:
                        coord = self.__normalize_coordinates(kps[key])
                        self.kps_dict[key].append(coord)
                    self.__detect_kps()

                img = cv2.resize(img, (1960,1080))
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


if __name__ == '__main__':
    model_path = "6_network/3_class_30_0626/model/TCN_struct6_2020-06-26-15-27-41.pth"

    # video = "1_video/swim/1_1_Trim.mp4"
    # Tester(model_path, video).test()

    #video_folder = "7_test/test/"
    #video_folder = "7_test/test_carol/video/"
    video_folder = "7_test/test/video/"
    for video in os.listdir(video_folder):
        Tester(model_path, os.path.join(video_folder, video)).test()

