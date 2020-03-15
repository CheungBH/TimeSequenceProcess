import torch
import sys
sys.path.append("../../")
from src.TCN.TCNsrc.model import TCN
import numpy as np
from config import config
import os

device = config.device
TCN_params = config.TCN_structure
kps_num = config.test_kps_num
cls = ["drown", "swim"]


class TCNPredictor:
    def __init__(self, model_name, n_classes):
        structure_num = int((model_name.split("/")[-1]).split('_')[1][6:])
        [channel_size, kernel_size, dilation] = TCN_params[structure_num]
        self.model = TCN(input_size=kps_num,
                         output_size=n_classes,
                         num_channels= channel_size,
                         kernel_size=kernel_size,
                         dropout=0,
                         dilation=dilation)
        self.model.load_state_dict(torch.load(model_name, map_location=device))
        if device != "cpu":
            self.model.cuda()
        self.model.eval()

    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=device)#(1, 30, 34)
        data = data.permute(0,2,1)
        return data#(1, 34, 30)

    def predict(self, data):
        input = self.get_input_data(data)
        output = self.model(input)
        #print('output:',output)
        pred = output.data.max(1, keepdim=True)[1]
        #print('pred:',pred)
        return pred


class TestWithtxt:
    def __init__(self, model_name):
        self.tester = TCNPredictor(model_name, 2)

    def pred_txt(self, txt_path):
        inps = np.loadtxt(txt_path)
        for inp in inps:
            inp = inp.astype(np.float32).reshape(-1, 34)
            res = self.tester.predict(inp)
            print(res)


if __name__ == '__main__':
    model_pth = "model/TCN_struct3_2020-03-15-09-34-17.pth"
    input_folder = 'data/drown'

    Test = TestWithtxt(model_pth)
    for txt in os.listdir(input_folder):
        print("Processing {}".format(txt))
        Test.pred_txt(os.path.join(input_folder, txt))
