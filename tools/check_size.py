import numpy as np

txt_path = '../4_data/cut_point/30frames/10steps/test100/test100_0.txt'

file_matrix = np.loadtxt(txt_path)
print("The file has {} rows".format(file_matrix.shape[0]))
try:
    print("Each row has {} items".format(file_matrix.shape[1]))
except IndexError:
    print("Each row has 1 items")
