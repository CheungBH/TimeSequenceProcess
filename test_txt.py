from src.TCN.test_TCN import TCNPredictor
# from src.LSTM.test_LSTM import LSTMPredictor
from src.ConvGRU.test_ConvGRU import ConvGRUPredictor
from src.BiLSTM.test_BiLSTM import BiLSTMPredictor
from collections import defaultdict
from src.ConvLSTM.test_ConvLstm import ConvLSTMPredictor
import numpy as np
import os
import csv


class txtTester:
    def __init__(self, data_src, model_path, cls):
        self.cls = cls
        self.data = np.loadtxt(os.path.join(data_src, "data.txt"))
        self.label = np.loadtxt(os.path.join(data_src, "label.txt"))
        self.model_path = model_path.split("/")[-1]
        self.tester = self.__get_tester(model_path)
        self.cnt = defaultdict(int)
        self.correct = defaultdict(int)

    def __get_tester(self, model):
        if "ConvLSTM" in model:
            return ConvLSTMPredictor(model, len(self.cls))
        if "BiLSTM" in model:
            return BiLSTMPredictor(model, len(self.cls))
        if "ConvGRU" in model:
            return ConvGRUPredictor(model, len(self.cls))
        # if 'LSTM' in model:
        #     return LSTMPredictor(model)
        if "TCN" in model:
            return TCNPredictor(model, len(self.cls))

    def start(self):
        cnt, correct, final_res = 0, 0, [self.model_path]
        for d, l in zip(self.data, self.label):
            cnt += 1
            self.cnt[self.cls[int(l)]] += 1
            res = self.tester.predict(d.reshape(30,34).astype(np.float32))
            if res == l:
                correct += 1
                self.correct[self.cls[int(l)]] += 1
            print("Prediction of sample {} is {} ------> {}".format(cnt, self.cls[res], (res == l).tolist()[0][0]))
        print("Total acc is {}".format(round(correct/cnt, 2)))
        final_res.append(round(correct/cnt, 2))
        for (k1,v1),(k2,v2) in zip(self.cnt.items(), self.correct.items()):
            print("The acc of {} is {}".format(k1, round(v2/v1, 2)))
            final_res.append(round(v2/v1, 2))
        return final_res


class txtAutoTesting:
    def __init__(self, model_folder, data_src):
        self.models = [os.path.join(model_folder, model_name) for model_name in os.listdir(model_folder)]
        self.data = data_src
        with open(os.path.join(data_src, "cls.txt"), "r") as cf:
            self.cls = []
            for line in cf.readlines():
                if "\n" in line:
                    self.cls.append(line[:-1])
                else:
                    self.cls.append(line)
        title = ["TRAINING SAMPLE RESULT","overall acc"]
        for cls in self.cls:
            title.append("{} acc".format(cls))
        self.content = [title]

    def process(self):
        for model in self.models:
            test = txtTester(self.data, model, self.cls)
            self.content.append(test.start())

    def write_csv(self, res_file):
        with open(res_file, "w", newline="") as f:
            csv_writer = csv.writer(f)
            for item in self.content:
                csv_writer.writerow(item)


if __name__ == '__main__':
    # data = '5_input/input1'
    # model = '6_network/net2/model/TCN_struct8_2020-03-19-12-26-23.pth'
    # txtT = txtTester(data, model, ["drown", "swim"])
    # result = txtT.start()
    # print(result)

    models = "tmp/net1/model"
    data = "5_input/input1"
    output = "tmp/train_data_res.csv"
    tAT = txtAutoTesting(models, data)
    tAT.process()
    tAT.write_csv(output)
