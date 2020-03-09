from models.TCN.train_TCN import TCNTrainer
from models.LSTM.train_LSTM import LSTMTrainer
from models.ConvLSTM.train_ConvLSTM import ConvLSTMTrainer
from config import config
import time
import shutil
import os

res_dest = config.out_dest
#if os.path.exists(res_dest): raise ValueError("The destination path '{}' exists!".format(res_dest))
os.makedirs(os.path.join(res_dest, "model"), exist_ok=True)
os.makedirs(os.path.join(res_dest, "log"), exist_ok=True)
n_classes = len(config.training_labels)


if __name__ == '__main__':
    res = open(os.path.join(res_dest, "result.txt"), "w")
    shutil.copy(os.path.join(config.data_path, "cls.txt"), res_dest)
    with open(os.path.join(res_dest, "cls.txt"), "a+") as f:
        f.write("\n" + config.data_info)
    res.write("model_name,model_type,epochs,dropout,learning-rate,min_train_loss,min_val_loss,max_val_acc\n")

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
                        train_loss, val_loss, val_acc = TCNTrainer(config.data_path, epoch, dropout, lr, model_name,
                                                          log_name, batch_size, n_classes).train_tcn()
                        cost = time.time() - begin_time
                        print("Total time cost is {}s\n".format(cost))
                        res.write("{},{},{},{},{},{},{},{}\n".format(
                            "TCN_" + time_str + ".pth", net, epoch, dropout, lr, train_loss, val_loss, val_acc))
                    elif net == "LSTM":
                        log_name = os.path.join(res_dest, "log", "{}_".format(net) + time_str + ".csv")
                        model_name = os.path.join(res_dest, "model", "LSTM_" + time_str + ".h5")
                        os.makedirs(os.path.join(res_dest, "LSTM_graph/loss"), exist_ok=True)
                        os.makedirs(os.path.join(res_dest, "LSTM_graph/acc"), exist_ok=True)
                        train_loss, val_loss, val_acc = LSTMTrainer(config.data_path, epoch, dropout, lr, model_name,
                                    log_name, batch_size, n_classes).train_LSTM()
                        cost = time.time() - begin_time
                        print("Total time cost is {}s\n".format(cost))
                        res.write("{},{},{},{},{},{},{},{}\n".format(
                            "LSTM_" + time_str + ".h5", net, epoch, dropout, lr, train_loss, val_loss, val_acc))
                    elif net == "ConvLSTM":
                        log_name = os.path.join(res_dest, "log", "{}_".format(net) + time_str + ".txt")
                        model_name = os.path.join(res_dest, "model", "ConvLSTM_" + time_str + ".pth")
                        train_loss, val_loss, val_acc = ConvLSTMTrainer(config.data_path, epoch, dropout, lr, model_name,
                                                               log_name, batch_size, n_classes).train_convlstm()
                        cost = time.time() - begin_time
                        print("Total time cost is {}s\n".format(cost))
                        res.write("{},{},{},{},{},{},{},{}\n".format("ConvLSTM_" + time_str + ".pth",
                                    net, epoch, dropout, lr, train_loss, val_loss, val_acc))
                    else:
                        raise ValueError("Wrong model type")
