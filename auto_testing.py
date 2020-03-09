import os
import cv2
from models.ConvLSTM.test_ConvLstm import ConvLSTMPredictor
from models.TCN.test_TCN import TCNPredictor
from models.LSTM.test_LSTM import LSTMPredictor


class TestVideo:
    def __init__(self, model, video, label):
        self.tester = self.__get_tester(model)
        self.label_dict = self.__read_label(label)
        self.cap = cv2.VideoCapture(video)
        self.result = []

    def __get_tester(self, model):
        if "LSTM" in model:
            return LSTMPredictor(model)
        elif 'ConvLSTM' in model:
            return ConvLSTMPredictor(model)
        elif "TCN" in model:
            return TCNPredictor(model)

    def __read_label(self, label):
        pass

    def process(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            if ret:
                pass
            else:
                break


class AutoTesting:
    def __init__(self, model_folder, video_folder, label_folder, result_path):
        self.model_ls = [os.path.join(model_folder, model) for model in os.listdir(model_folder)]
        self.label_ls = [os.path.join(label_folder, label) for label in os.listdir(label_folder)]
        self.video_ls = [os.path.join(video_folder, label) for label in os.listdir(video_folder)]
        self.result = [self.__get_label_title()]
        self.result_path = result_path

    def __get_label_title(self):
        title = []
        for label_file in self.label_ls:
            with open(label_file, "r") as lf:
                pass
        return title

    def test(self):
        for m in self.model_ls:
            results = []
            for v, l in zip(self.video_ls, self.label_ls):
                res = TestVideo(m, v, l).process()
                results += (res)
            self.result.append(results)
        self.__write_result()

    def __write_result(self):
        pass

