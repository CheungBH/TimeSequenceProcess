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
video_process_class = ["drown"]

# Coordinate process
coord_step = 10
coord_frame = 30
coord_process_method = "ordinary"  #Do not change now
coord_process_class = ["normal", "drown"]

# Merge input
merge_step = 10
merge_frame = 30
merge_process_method = "ordinary"
merge_class = ["normal", "drown"]
merge_dest_name = "input1"
merge_comment = "This is a test"

# Auto training config
batch_size = 32
activation = 'relu'
optimizer = 'Adam'
data_path = 'network'

epoch_ls = [100]
dropout_ls = [0.1]
network_structure_ls = [0]
val_ratio_ls = [0.2]

class_name = ["Backswing", "Standing", "Final", "Downswing"]
X_vector = 36
training_frame = 15
data_info = "test for openpose"
begin_num = 1




