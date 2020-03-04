import numpy as np

file = "../tmp/test/data.txt"

f = open(file, "r")

data = []
for line in f.readlines():
    origin_ls = line.split("\t")
    try:
        origin_ls.remove("\n")
    except:
        pass
    while True:
        try:
            origin_ls.remove("")
        except ValueError:
            break
    ls = [float(item) for item in origin_ls]
    n = np.array(ls).reshape((30, 34))
    data.append(n)
    print(n)
    a = 1
    # print(np.array([line]).reshape((30, 34)) + "\n")

print(data)
#
# txt = [[1,2,3,2,3,4,3,4,5,4,5,6]]
# res = np.array(txt).reshape((4,3))   #(row, column)
#
# print(res)



