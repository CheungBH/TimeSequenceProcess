import torch
import sys
sys.path.append("../../")
from src.TCN.TCNsrc.model import TCN
import numpy as np
from config import config

device = config.device
params_ls = config.TCN_structure


class TCNPredictor:
    def __init__(self, model_name):
        structure_num = int(model_name.split('_')[1][6:])
        self.model = TCN(input_size=34,
                         output_size=2,
                         num_channels= params_ls[structure_num][0],
                         kernel_size=params_ls[structure_num][1],
                         dropout=0)
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


if __name__ == '__main__':
    model_pth = 'TCN_struct1_10-10-10-10-10-10.pth'
    input_pth = 'data.txt'
    inp = np.loadtxt(input_pth).astype(np.float32).reshape(-1,34)
    prediction = TCNPredictor(model_pth)
    res = prediction.predict(inp)
    print(res)