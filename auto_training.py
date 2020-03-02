from models.TCN.train_TCN import TCNTrainer
from models.LSTM.train_LSTM import LSTMTrainer
from config import config
import time
import shutil
import os

res_dest = config.out_dest
os.makedirs(os.path.join(res_dest, "model"), exist_ok=True)
os.makedirs(os.path.join(res_dest, "log"), exist_ok=True)


if __name__ == '__main__':
    res = open(os.path.join(res_dest, "result.txt"), "w")
    shutil.copy(os.path.join(config.data_path, "cls.txt"), res_dest)
    with open(os.path.join(res_dest, "cls.txt"), "a+") as f:
        f.write("\n" + config.data_info)
        f.close()

    cnt = 0
    for net in config.networks:
        for epoch in config.epoch_ls[net]:
            for dropout in config.dropout_ls[net]:
                for lr in config.lr_ls[net]:
                    cnt += 1
                    print("Begin training network {}".format(cnt))
                    print("Begin training {}: {} epochs, {} dropout, {} lr".format(net, epoch, dropout, lr))
                    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    begin_time = time.time()
                    batch_size = config.batch_size[net]
                    if net == "TCN":
                        log_name = os.path.join(res_dest, "log", "{}_".format(net) + time_str + ".txt")
                        model_name = os.path.join(res_dest, "model", "TCN_" + time_str + ".pth")
                        train_loss, val_loss = TCNTrainer(config.data_path, epoch, dropout, lr, model_name, log_name, batch_size).train_tcn()
                        cost = time.time() - begin_time
                        print("Total time cost is {}s\n".format(cost))
                        res.write("{}: \n{} epochs, {} learning-rate, {} dropout\n".format("TCN_" + time_str + ".pth", epoch, lr, dropout))
                        res.write("Min train loss is {}. Min val loss is {}\n\n".format(train_loss, val_loss))
                    elif net == "LSTM":
                        log_name = os.path.join(res_dest, "log", "{}_".format(net) + time_str + ".csv")
                        model_name = os.path.join(res_dest, "model", "LSTM_" + time_str + ".h5")
                        os.makedirs(os.path.join(res_dest, "LSTM_graph/loss"), exist_ok=True)
                        os.makedirs(os.path.join(res_dest, "LSTM_graph/acc"), exist_ok=True)
                        LSTMTrainer(config.data_path, epoch, dropout, lr, model_name, log_name, batch_size).train_LSTM()
                        cost = time.time() - begin_time
                        print("Total time cost is {}s\n".format(cost))
                        res.write("{}: \n{} epochs, {} learning-rate, {} dropout\n\n".format("LSTM_" + time_str + ".pth", epoch, lr, dropout))
                        # res.write("Min train loss is {}. Min val loss is {}\n\n".format(train_loss, val_loss))
                    else:
                        raise ValueError("Wrong model type")
