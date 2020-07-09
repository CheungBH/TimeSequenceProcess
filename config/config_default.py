import torch

device = "cuda:0"
print("Using {}".format(device))
size = (540, 360)

# For yolo
confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

# yolo_cfg = "src/yolo/cfg/yolov3-swim-416.cfg"
# yolo_weights = 'weights/yolo/yolov3-swim-416_40000.weights'
yolo_cfg = "config/yolo_cfg/yolov3-spp-1cls.cfg"
yolo_weight = 'weights/yolo/best.weights'

# For SPPE
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80

pose_backbone = "seresnet101"
pose_weight = './weights/sppe/duc_se.pth'
pose_cfg = None

body_parts_ls = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
body_part = {value: idx for idx, value in enumerate(body_parts_ls)}
pose_cls = 17


# TCN structure
TCN_structure = {1:[[6, 6, 6, 6], 5, 2],
                 # [channel_size, kernel_size, dilation]
                 2: [[3, 4, 5, 6], 5, 4],
                 3: [[6, 6, 6, 6], 7, 2],
                 4: [[12, 12, 12, 12], 7, 2],
                 5: [[8, 16, 8, 16], 7, 2],
                 6: [[6, 6, 6, 6], 7, 4],
                 7: [[6, 6, 6, 6], 7, 8],
                 8: [[6, 6, 6, 6], 7, 1]
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


TCN_single = True
# Video process
video_process_class = ["trash"]
save_kps_video = False
save_kps_img = True
save_black_img = False
save_frame = False
process_gray = True


# Coordinate process
coord_step = 5
coord_frame = 30
coord_process_method = "ordinary"  #Do not change now
coord_process_class = ["drown_0624", "Stand", "swim_0623"]



# Merge input
merge_step = 5
merge_frame = 30
merge_process_method = "ordinary"
merge_class = ["drown_0624", "Stand"]
merge_dest_name = "2_class_30_0707"
merge_comment = "{}: [drown_0624,Stand], all samples, 30f, 5s, wrong posture has been deleted".format(merge_dest_name)


# Auto training config
train_data_path = '5_input/2_class_30_0707'
out_dest = "6_network/3_class_30_0707"
data_info = "{}: The data comes from {}, equal data, [drown_0624,Stand], 30 frames, 5 steps".format(out_dest.split("/")[1],
                                                                      "/".join(train_data_path.split("/")[1:]))
network_name = ["ConvLSTM", "ConvGRU", "BiLSTM", "TCN", "LSTM"]
network_num = [3,4]


networks = [network_name[idx] for idx in network_num]
epoch_ls = {"LSTM": [80],
            "TCN": [500],
            "ConvLSTM": [100],
            "ConvGRU": [100],
            "BiLSTM": [300], }
dropout_ls = {"LSTM": [0.05],
              "TCN": [0.05],
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
    "ConvLSTM": [1, 2, 3],
    "LSTM": [4,6], #checked [4, 5, 6, 7, 8]
    "TCN": [1,4,5,6,7], #checked 2
    "ConvGRU": [1, 2],
    "BiLSTM": [1, 2],
}

batch_size = {"LSTM": 32, "TCN": 32, "ConvLSTM": 8, "ConvGRU": 8, "BiLSTM": 128} #{"LSTM": 128, "TCN": 128, "ConvLSTM": 64, "ConvGRU": 128, "BiLSTM": 128}
kps_num = 34
training_frame = 30
log_interval = 10
train_val_ratio = 0.2
training_labels = {0: "drown", 1: "Stand", 2:"swim" }


# Auto labelling config
label_comment = "label1: 30 frames, 3 classes: (drown,Stand,swim)"
label_frame = 30
label_cls = ["drown_side", "Stand","swim" ]
label_folder_name = "trash"
label_main_folder = "7_test/test"
write_label_info = True


# Auto testing config
test_model_folder = "6_network/3_class_30_0626_more/model"
test_video_folder = "7_test/test/video"
test_label_folder = "7_test/test/trash"
test_res_file = "8_result/trash/trash.csv"
test_write_video = True

test_kps_num = 34
testing_frame = 30

testing_log = "{}: all the datas, swim and drown, 30 frames, 10 steps. Video coming from v_1, and label_1".\
    format(test_res_file.split("/")[1])
