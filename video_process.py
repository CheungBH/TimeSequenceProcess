from src.human_detection import ImgProcessor
import cv2
from config.config import video_process_class
import os

IP = ImgProcessor()


class VideoProcessor:
    def __init__(self, video_path, draw_video_path, output_txt_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.coord = []
        self.out = cv2.VideoWriter(draw_video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (self.height, self.width))
        self.file = open(output_txt_path, "w")

    def __normalize_coordinates(self, coordinates):
        for i in range(len(coordinates)):
            if (i+1) % 2 == 0:
                coordinates[i] = coordinates[i] / self.height
            else:
                coordinates[i] = coordinates[i] / self.width
        return coordinates

    def __write_txt(self):
        for item in self.coord:
            self.file.write(str(item)+"\t")
        self.file.write("\n")

    def process_video(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                kps, img = IP.process_img(frame)
                self.coord = self.__normalize_coordinates(kps)
                cv2.imshow("res", img)
                cv2.waitKey(2)
                self.out.write(cv2.resize(img, (self.width, self.height)))
                self.__write_txt()
            else:
                self.cap.release()
                self.file.close()
                self.out.release()
                break


class VideoFolderProcessor:
    def __init__(self, folder):
        self.video_ls = [os.path.join("1_video", folder, video_name) for video_name in os.path.join(os.path.join("1_video", folder))]
        self.drawn_video_ls = [path_name.replace("1_video", "2_drawn_video") for path_name in self.video_ls]
        self.txt_ls = [path_name.replace("1_video", "3_coord") for path_name in self.video_ls]
        os.makedirs(os.path.join("2_drawn_video", folder), exist_ok=True)
        os.makedirs(os.path.join("3_coord", folder), exist_ok=True)
        print("Processing video folder: {}".format(folder))

    def process_folder(self):
        for idx in range(len(self.video_ls)):
            VP = VideoProcessor(self.video_ls[idx], self.drawn_video_ls[idx], self.txt_ls[idx])
            VP.process_video()


if __name__ == '__main__':
    VP = VideoProcessor("test/normal_2.avi", "test/normal_2_kps.avi", "test/normal_2.txt")
    VP.process_video()

    for cls in video_process_class:
        VFP = VideoFolderProcessor(cls)
        VFP.process_folder()
