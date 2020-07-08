from collections import defaultdict


class KPSProcessor:
    def __init__(self, h, w):
        self.height = h
        self.width = w

    def __normalize_coordinates(self, coordinates):
        for i in range(len(coordinates)):
            if (i + 1) % 2 == 0:
                coordinates[i] = coordinates[i] / self.height
            else:
                coordinates[i] = coordinates[i] / self.width
        return coordinates

    def __select_kps(self, idx, kps):
        if idx in kps.keys():
            return kps[idx]
        else:
            return [[]]

    def __dim2to1(self, raw_kp):
        kp = []
        for pt in raw_kp:
            for dim in pt:
                kp.append(dim)
        return kp

    def __dictdim2to1(self, raw_kp_dict):
        kps_dict = defaultdict(list)
        for k, v in raw_kp_dict.items():
            kp = self.__dim2to1(v)
            kps_dict[k] = kp
        return kps_dict

    def process_single_kps(self, kps, idx):
        kp = self.__select_kps(idx, kps)
        kp = self.__dim2to1(kp)
        coord = self.__normalize_coordinates(kp)
        return coord

    def process_kp(self, kp):
        kp = self.__dim2to1(kp)
        coord = self.__normalize_coordinates(kp)
        return coord



