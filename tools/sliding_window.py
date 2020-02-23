

# with open("../tmp/normal/normal_0.txt", "r") as f:
#     # data = f.read()
#
#     lines = [line for line in f.readlines()]
#
# #
# # print(lines)
#
# print(list(range(60)[::5]))

frame, step = 30, 10


def process( coord_path, dest_path):
    with open(coord_path, "r") as rf:
        lines = [line for line in rf.readlines()]
        rf.close()

    outs = []
    try:
        for begin_idx in range(len(lines))[::step]:
            out = ""
            for idx in range(frame):
                out += lines[begin_idx+idx].replace("\n", "\t")
            outs.append(out[:-1] + '\n')
    except IndexError:
        pass

    # outs = [lines[idx: idx + frame] for idx in range(len(lines))[::step]]
    a = 1
    with open(dest_path, "w") as wf:
        for out in outs:
            wf.write(out)


if __name__ == '__main__':
    process("../tmp/normal/normal_0.txt", "../tmp/normal/normal_0_out.txt")
