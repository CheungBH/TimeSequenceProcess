import numpy as np
from keras.models import load_model
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


class LSTMPredictor:
    def __init__(self, model_pth):
        self.model = load_model(model_pth)

    def predict(self, data):
        inp = data[np.newaxis, ]
        output = self.model.predict(inp)
        return np.argmax(output)


if __name__ == '__main__':
    input_pth = 'data/data.txt'
    input_data = np.loadtxt(input_pth)
    model_pth = "models/model.h5"

    # model = load_model("models/model.h5")
    # res = model.predict(input_data)
    # print(res)

    predictor = LSTMPredictor(model_pth)
    res = predictor.predict(input_data)
    print(res)
