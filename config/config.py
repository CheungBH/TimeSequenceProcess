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


body_parts_ls = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_part = {value: idx for idx, value in enumerate(body_parts_ls)}

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
video_process_class = ["drown", "swim"]



# Coordinate process
coord_step = 5
coord_frame = 30
coord_process_method = "ordinary"  #Do not change now
coord_process_class = ["drown", "swim"]



# Merge input
merge_step = 5
merge_frame = 30
merge_process_method = "ordinary"
merge_class = ["drown", "swim"]
merge_dest_name = "input4"
merge_comment = "{}: [drown, swim], all samples, 20f, 10s, wrong posture has been deleted".format(merge_dest_name)



# Auto training config
train_data_path = '5_input/input4'
networks = ["BiLSTM"]
out_dest = "6_network/nettest"
data_info = "{}: The data comes from {}, all data, [drown, swim], 30 frames, 3 steps".format(out_dest.split("/")[-1], train_data_path.split("/")[-1])

epoch_ls = {"LSTM": [2],
            "TCN": [200, 100],
            "ConvLSTM": [50],
            "ConvGRU": [50],
            "BiLSTM": [80], }
dropout_ls = {"LSTM": [0.2],
              "TCN": [0.05, 0.1],
              "ConvLSTM": [""],
              "ConvGRU": [""],
              "BiLSTM": [""]
              }  # ConvLSTM/ConvGRU/BiLSTM don't have any dropouts, do not change
lr_ls = {"LSTM": [1e-4],
         "TCN": [1e-4],
         "ConvLSTM": [1e-4],
         "ConvGRU": [1e-4],
         "BiLSTM": [1e-4], }
structure_ls = {
    "ConvLSTM": [1, 2, 3, ],
    "LSTM": [4, 5, 6, 7, 8],
    "TCN": [1, 2, 3, 4, 5],
    "ConvGRU": [1, 2, 3],
    "BiLSTM": [1, 2, 3],
}

batch_size = {"LSTM": 128, "TCN": 128, "ConvLSTM": 64, "ConvGRU": 128, "BiLSTM": 128}
kps_num = 34
training_frame = 30
log_interval = 5
train_val_ratio = 0.2
training_labels = {0: "drown", 1: "swim", }



# Auto labelling config
label_comment = "label1: 30 frames, 2 classes: (swim, drown)"
label_frame = 30
label_cls = ["drown", "swim", ]
label_folder_name = "label1"
label_main_folder = "7_test/test_v"



# Auto testing config
test_model_folder = "6_network/net1/model"
test_video_folder = "7_test/train_v/video"
test_label_folder = "7_test/train_v/label1"
test_res_file = "8_result/result4/train_video_res.csv"

test_kps_num = 34
testing_frame = 30

test_comment = "{}: all the datas, swim and drown, 30 frames, 10 steps. Video coming from v_1, and label_1".\
    format(test_res_file.split("/")[1])
