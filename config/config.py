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
                 # [channel_size, kernel_size, dilation]
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

# ConvGRU structure
ConvGRU_structure = {1: [[128, 64, 64, 32, 32], (5,5), False],
                     2: [[128, 64, 64, 32, 32], (5,5), True],
                     }

# BiLSTM structure
BiLSTM_structure = {1: [64, 2, False],
                    2: [64, 2, True],
                    }


# Video process
video_process_class = ["drown_new", "swim_new"]



# Coordinate process
coord_step = 10
coord_frame = 30
coord_process_method = "ordinary"  #Do not change now
coord_process_class = ["drown_new", "swim_new"]



# Merge input
merge_step = 10
merge_frame = 30
merge_process_method = "ordinary"
merge_class = ["drown_new", "swim_new"]
merge_dest_name = "2_class_new"
merge_comment = "Input test: swim, drown_new, swim_new, all samples, 30f, 10s"



# Auto training config
num_classes_pose = 2
train_data_path = '5_input/2_class_new'
networks = ["ConvGRU", "BiLSTM", "TCN", "ConvLSTM", "LSTM"]
out_dest = "6_network/2_class_b8_new"
data_info = "net1: The data comes from input1, all data, label is {swim, drown}, 30 frames, 10 steps"

epoch_ls = {"LSTM": [500],
            "TCN": [500],
            "ConvLSTM": [300],
            "ConvGRU": [300],
            "BiLSTM": [300], }
dropout_ls = {"LSTM": [0.1],
              "TCN": [0.05, 0.1],
              "ConvLSTM": [0],
              "ConvGRU": [0],
              "BiLSTM": [0]
              }  # ConvLSTM/ConvGRU/BiLSTM don't have any dropouts, do not change
lr_ls = {"LSTM": [1e-4],
         "TCN": [1e-4],
         "ConvLSTM": [1e-4],
         "ConvGRU": [1e-4],
         "BiLSTM": [1e-4], }
structure_ls = {
    "ConvLSTM": [6],
    "LSTM": [6, 7, 8],
    "TCN": [6, 7, 8],
    "ConvGRU": [1],
    "BiLSTM": [1],
}

batch_size = {"LSTM": 8, "TCN": 8, "ConvLSTM": 8, "ConvGRU": 8, "BiLSTM": 8}
kps_num = 34
training_frame = 30
log_interval = 5
train_val_ratio = 0.2
training_labels = {0: "drown", 1: "swim" }



# Auto labelling config
label_comment = "label1: 30 frames, 2 classes: (swim, drown)"
label_frame = 30
label_cls = ["drown", "swim", ]
label_folder_name = "label_2class"
label_main_folder = "7_test/test_v"



# Auto testing config
test_model_folder = "6_network/2_class_b8_new/model/"
test_video_folder = "7_test/test_1/video/"
test_label_folder = "7_test/test_1/label1/"
test_res_file = "8_result/result4/train_video_res.csv"

test_kps_num = 34
testing_frame = 30
testing_log = "{}: all the datas, swim and drown, 30 frames, 10 steps. Video coming from v_1, and label_1".\
    format(test_res_file.split("/")[1])

test_comment = "result1: all the datas, swim and drown, 30 frames, 10 steps. Video coming from v_1, and label_1"
