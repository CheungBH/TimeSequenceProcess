import numpy as np
from keras.models import load_model


class LSTMPredictor:
    def __init__(self, model_pth):
        self.model = model_pth

    def predict(self, data):
        output = self.model.predict(data)
        return np.argmax(output)


if __name__ == '__main__':
    pass
    # input_pth = 'data/AlphaPose_drown_34.txt'
    # input_data = np.loadtxt(input_pth)
    # input_data = input_data[np.newaxis,]
    #
    #
    # model = load_model("model.h5")
    # output = model.predict(input_data)
    # pred = np.argmax(output)
    # print(pred)