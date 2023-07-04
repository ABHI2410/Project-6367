import cv2
import os
import pandas as pd
from tqdm import tqdm
# from skimage.io import Video


from KeyPointDetector import KeyPointDetector

class Convertor:
    def __init__(self) -> None:

        directory = os.getcwd()
        os.chdir("..")
        os.chdir("Dataset/raw_videos/")
        self.video_location = os.getcwd()


        os.chdir("../processed_videos")
        self.procesed_video_location = os.getcwd()
        os.chdir("../raw_data/")
        self.csv_location = os.getcwd() + '/' + "how2sign_realigned_train.csv"
        os.chdir("../../VideoProcessing")

    def video_clipping(self):
        df = pd.read_csv(self.csv_location, sep= "	", header= 0)
        for row in df.itertuples(index=True, name='Pandas'):
            video_name = row.VIDEO_NAME
            start_time = row.START_REALIGNED
            end_time = row.END_REALIGNED
            from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
            ffmpeg_extract_subclip(self.video_location +'/'+ video_name + ".mp4", start_time, end_time, targetname=f"{self.procesed_video_location}/{video_name}{str(start_time).replace('.', '')}{str(end_time).replace('.', '')}.mp4")


    def process_video(self,video_name):
        # input = Video(self.video_location + video_name)
        # fc = input.frame_count()
        # print(fc)
        input = cv2.VideoCapture(self.procesed_video_location + '/' + video_name)
        os.remove(self.procesed_video_location + '/' + video_name)
        output = cv2.VideoWriter(
            self.procesed_video_location + '/' + video_name,
            cv2.VideoWriter_fourcc('m','p','4','v'),
            30, 
            (1280,720)
            )
        print("Entering Loop")
        while input.isOpened():
            _i , frame = input.read()
            if (_i):
                obj = KeyPointDetector()
                frame_output = obj.keypoints(frame)
                del obj
                output.write(frame_output)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        output.release()
        input.release()
        cv2.destroyAllWindows()

