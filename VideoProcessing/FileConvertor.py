import cv2
import os
import pandas as pd
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from KeyPointDetector import KeyPointDetector
from multiprocessing import Pool

class Convertor:
    def __init__(self) -> None:

        directory = os.getcwd()
        os.chdir("../../../../../../../d")
        os.chdir("Project-6367/Dataset/raw_videos")
        self.video_location = os.getcwd()
        os.chdir("../clip_videos")
        self.video_clip_location = os.getcwd()
        os.chdir("../raw_data/")
        self.csv_location = os.getcwd() + '/' + "how2sign_realigned_train.csv"
        os.chdir("../../../../c/Users/patel/OneDrive/Documents/Project-6367")

        # os.makedirs("processed_videos")
        os.chdir("processed_videos")
        self.procesed_video_location = os.getcwd()
        os.chdir("../")
        # os.makedirs("clip_videos")

        os.chdir("../VideoProcessing")

    def video_clipping(self):
        df = pd.read_csv(self.csv_location, sep= "	", header= 0)
        def clip(row):
            video_name = row.VIDEO_NAME
            start_time = row.START_REALIGNED
            end_time = row.END_REALIGNED
            ffmpeg_extract_subclip(self.video_location +'/'+ video_name + ".mp4", start_time, end_time, targetname=f"{self.video_clip_location}/{video_name}{str(start_time).replace('.', '')}{str(end_time).replace('.', '')}.mp4")
        with Pool(processes=4) as pool:
            tqdm(pool.imap(clip,df.itertuples(index=True, name='Pandas')))


    def video_2_frame(self,video_name):
        try:
            if os.path.isfile(self.procesed_video_location + '/' + video_name.split('.')[0]):
                # print("Proccess completed once before for file: ",video_name)
                return 0
            input = cv2.VideoCapture(self.video_clip_location + '/' + video_name)
            if not input.isOpened():
                return
            os.makedirs(self.procesed_video_location+'/'+video_name.split('.')[0])
            base_path = self.procesed_video_location+'/'+ video_name.split('.')[0]
            digit = len(str(int(input.get(cv2.CAP_PROP_FRAME_COUNT))))
            n=0
            while True:
                _i , frame = input.read()
                if (_i):
                    cv2.imwrite(f'{base_path}_{str(n).zfill(digit)}.jpg',frame)
                    n += 1
                    if cv2.waitKey(1) == ord('q'):
                        break
                else:
                    break
            input.release()
            cv2.destroyAllWindows()
            # print("Completed key point detection for file: ", video_name)
            return 0
        except Exception as e:
            print(e)
            print("This video had error processing: ",video_name)
            return 0 
