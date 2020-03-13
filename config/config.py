import torch

device = "cuda:0"
print("Using {}".format(device))
size = (540, 360)

# For yolo
confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

# For SPPE
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80


# TCN structure
TCN_structure = {1:[[6, 6, 6, 6], 5, 2],
                 # [channel_size, kernel_size]
                 2: [[3, 4, 5, 6], 5, 4],
                 3: [[6, 6, 6, 6], 7, 2],
                 4: [[12, 12, 12, 12], 7, 2],
                 5: [[8, 16, 8, 16], 7, 2],
                 6: [[6, 6, 6, 6], 7, 4],
                 7: [[6, 6, 6, 6], 7, 8],
                 8: [[6, 6, 6, 6], 7, 1],
                 }


# ConvLSTM structure
ConvLSTM_structure = {1: [[128, 64, 64, 32, 32], (7, 7), False],
                      # [hidden_channels, kernel_size, attention]
                      2: [[128, 64, 64, 32, 32], (7, 7), True],
                      3: [[128, 64, 64, 64, 32], (5, 5), True],
                      4: [[128, 64, 32], (3, 3), True],
                      5: [[128, 64, 32, 32, 32], (5, 5), True],
                      6: [[128, 64, 32], (3, 3), False],
                      7: [[128, 128, 128, 64, 32], (7, 7), False],
                      8: [[128, 128, 128, 64, 32], (7, 7), True],
                      9: [[128, 128, 128, 64, 32], (3, 3), True],
                      10: [[128, 32], (3, 3), False],
                      }


# LSTM structure
LSTM_structure = {1: [[128, 128], [64, 16]],
                  2: [[128, 128, 128], [64, 32, 8]],
                  3: [[128, 128], [16]],
                  4: [[128, 64], [64, 8]],
                  5: [[128, 128, 128, 128], [64, 16]],
                  6: [[128, 128, 128], [64, 16]],
                  7: [[128, 128], [64, 32, 16, 8]],
                  8: [[128, 64], [64, 64, 16]],
                  9: [[128, 128, 128], [64, 32, 16, 8]],
                  }

# Video process
video_process_class = ["drown", "swim"]



# Coordinate process
coord_step = 10
coord_frame = 20
coord_process_method = "ordinary"  #Do not change now
coord_process_class = ["test100", "test101"]



# Merge input
merge_step = 10
merge_frame = 30
merge_process_method = "ordinary"
merge_class = ["swim", "test100", "test101"]
merge_dest_name = "input_test"
merge_comment = "Inpute test: swim, test100, test101, all samples, 30f, 10s"



# Auto training config
train_data_path = '5_input/input1/'
networks = ["LSTM"]

epoch_ls = {"LSTM": [200, 500, 1000],
            "TCN": [100, 200, 500],
            "ConvLSTM": [100, 200]}
dropout_ls = {"LSTM": [0.2],
              "TCN": [0.05, 0.1],
              "ConvLSTM": [""]}  # ConvLSTM don't have any dropouts
lr_ls = {"LSTM": [1e-4],
         "TCN": [1e-4],
         "ConvLSTM": [1e-4]}
structure_ls = {
    "ConvLSTM": [1,2,3,4,5,6,7,8],
    "LSTM": [1,2,3,4,5,6,7,8,9],
    "TCN": [1,2,3,4,5,6,7,8,9,10],
}

batch_size = {"LSTM": 128, "TCN": 128, "ConvLSTM": 32}
kps_num = 34
training_frame = 30
log_interval = 5
training_labels = {0:"swim", 1:"drown"}

data_info = "The data comes from input1, all datas, label is {swim, drown}, 30 frames, 10 steps"
out_dest = "6_network/test"



# Auto labelling config
label_comment = "label1: 30 frames, 2 classes: (swim, drown)"
label_frame = 30
label_cls = ["swim", "drown"]
label_folder_name = "label1"
label_video_src = ""



# Auto testing config
test_model_folder = ""
test_video_folder = ""
test_res_folder = ""
test_label_folder = ""

test_kps_num = 34
testing_frame = 30

test_comment = "Input "

