import os
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Masking, Dropout, BatchNormalization

from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, CSVLogger


batch_size = 16
input_channels = 30
seq_length = 34


class LSTMTrainer:
    def __init__(self, data_path, epoch, dropout, lr, model_name, log_name):
        self.dropout = dropout
        self.epoch = epoch
        self.data_path = data_path
        self.lr = lr
        self.model_name = model_name
        self.log_name = log_name
        self.model = self.get_model()

    def get_label(self, file_path):
        file_name = os.path.split(file_path)[-1]
        label = file_name.split('_')[1]
        if label == 'drown':
            return 1
        else:
            return 0

    def sequence_padding(self, array, maxlen, value):
        num_timesteps, num_dims = array.shape
        padding = value * np.ones((maxlen - num_timesteps, num_dims))
        new_array = np.concatenate([array, padding], axis=0)

        return new_array

    def load_dataset(self, input_dir):
        datas = []
        labels = []

        for file in os.listdir(input_dir):
            if file == '.ipynb_checkpoints':
                continue
            else:
                path = os.path.join(input_dir, file)
                data = np.loadtxt(path, ndmin=2)

                data = self.sequence_padding(data, maxlen=30, value=-1)  # 不够60行的话补-1，最长的frame是-1
                label = self.get_label(path)
                datas.append(data)
                labels.append(label)

        return np.array(datas), np.array(labels)

    def get_model(self):
        model = Sequential()

        model.add(Masking(mask_value=-1, input_shape=(input_channels, seq_length)))  # 此函数定义是，如果后面是-1就不参与计算

        # model.add(LSTM(16,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(128, dropout=self.dropout, recurrent_dropout=self.dropout, return_sequences=True))
        model.add(LSTM(128, dropout=self.dropout, recurrent_dropout=self.dropout, return_sequences=False))

        model.add(Dense(64, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def scheduler(self, epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.95)
        return K.get_value(self.model.optimizer.lr)

    def train_LSTM(self):
        reduce_lr = LearningRateScheduler(self.scheduler)
        filename = self.log_name
        csv_logger = CSVLogger(filename, separator=',', append=True)
        callbacks_list = [csv_logger, reduce_lr]
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        train_x, train_y = self.load_dataset(self.data_path)
        hist = self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=self.epoch, verbose=1,
                              callbacks=callbacks_list, validation_split=0.2)
        self.model.save(self.model_name)


if __name__ == '__main__':
    LSTMTrainer("data", 1000, 0.2, "", "model.h5", 'train_log.csv').train_LSTM()
