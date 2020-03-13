import torch
import sys
sys.path.append("../../")
from src.TCN.TCNsrc.model import TCN
import numpy as np
from config import config

device = config.device
TCN_params = config.TCN_structure
kps_num = config.test_kps_num


class TCNPredictor:
    def __init__(self, model_name, n_classes):
        structure_num = int(model_name.split('_')[1][6:])
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


if __name__ == '__main__':
    model_pth = 'TCN_struct1_10-10-10-10-10-10.pth'
    input_pth = 'data.txt'
    inp = np.loadtxt(input_pth).astype(np.float32).reshape(-1,34)
    prediction = TCNPredictor(model_pth, 2)
    res = prediction.predict(inp)
    print(res)