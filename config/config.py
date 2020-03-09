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
coord_frame = 20
coord_process_method = "ordinary"  #Do not change now
coord_process_class = ["test100", "test101"]

# Merge input
merge_step = 10
merge_frame = 30
merge_process_method = "ordinary"
merge_class = ["swim", "test100", "test101"]
merge_dest_name = "input_test"
merge_comment = "Inpute test: swim,test100,test101, all samples, 30f, 10s"

# Auto training config
data_path = '5_input/input1/'
networks = ["TCN", "ConvLSTM"]
epoch_ls = {"LSTM": [1],
            "TCN": [20, 10],
            "ConvLSTM": [5, 10]}
dropout_ls = {"LSTM": [0.2, 0.4],
              "TCN": [0.05, 0.1],
              "ConvLSTM": [""]}  # ConvLSTM don't have any dropouts
lr_ls = {"LSTM": [1e-4],
         "TCN": [1e-4],
         "ConvLSTM": [1e-4]}

data_info = "The data comes from input1"
batch_size = {"LSTM": 128, "TCN": 128, "ConvLSTM": 32}
out_dest = "tmp/mix3"

kps_num = 34
training_frame = 30
log_interval = 5
training_labels = {0:"swim",1:"drown"}


