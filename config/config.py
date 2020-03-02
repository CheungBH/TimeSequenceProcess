import torch

device = "cuda:0"
print("Using {}".format(device))

confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80


# Video process
video_process_class = ["drown", "swim"]

# Coordinate process
coord_step = 10
coord_frame = 30
coord_process_method = "ordinary"  #Do not change now
coord_process_class = ["swim", "drown"]

# Merge input
merge_step = 10
merge_frame = 30
merge_process_method = "ordinary"
merge_class = ["swim", "drown"]
merge_dest_name = "input1"
merge_comment = "Origin swim and drown, with 30 frames, 10 steps"

# Auto training config
data_path = '5_input/input1'
networks = ["LSTM"]
epoch_ls = {"LSTM": [10, 15], "TCN": [10, 20, 30]}
dropout_ls = {"LSTM": [0.2, 0.3], "TCN": [0.05, 0.1, 0.2]}
lr_ls = {"LSTM": [1e-4], "TCN": [1e-4]}

data_info = "The data comes from input1"
batch_size = {"LSTM": 128, "TCN": 128}
out_dest = "tmp/LSTM5"

kps_num = 34
training_frame = 30


