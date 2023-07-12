import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from KeyPointDetector import KeyPointDetector
from multiprocessing import Pool
from PIL import Image
import mediapipe as mp

# Colors 
WHITE = (225, 225, 225)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

class Convertor:
    holistic = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils
    face_spec = drawing.DrawingSpec(color=BLUE, thickness=1, circle_radius=1)
    pose_spec = drawing.DrawingSpec(color=WHITE, thickness=3, circle_radius=3)
    hand_spec = drawing.DrawingSpec(color=GREEN, thickness=3, circle_radius=3)
    dot_spec = drawing.DrawingSpec(color=RED, thickness=2, circle_radius=3)
    def __init__(self) -> None:
        directory = os.getcwd()
        os.chdir("..\\..\\..\\..\\")
        os.chdir("D:\\")
        os.chdir("Project-6367\\Dataset\\raw_videos")
        self.video_location = os.getcwd()
        os.chdir("..\\clip_videos")
        self.video_clip_location = os.getcwd()
        os.chdir("../raw_data/")
        self.csv_location = os.getcwd() + '\\' + "how2sign_realigned_train.csv"
        os.chdir("C:\\Users\\patel\\Project-6367")
        os.chdir("processed_videos")
        self.procesed_video_location = os.getcwd()
        os.chdir("..\\VideoProcessing")

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
        if os.path.exists(self.procesed_video_location + '\\' + video_name.split('.')[0]+'\\'):
            print("Proccess completed once before for file: ",video_name)
            return 0
        input = cv2.VideoCapture(self.video_clip_location + '\\' + video_name)
        output = cv2.VideoWriter(
            self.procesed_video_location + '\\' + video_name,
            cv2.VideoWriter_fourcc('m','p','4','v'),
            30, 
            (1280,720)
            )
        if not input.isOpened():
            return
        os.makedirs(self.procesed_video_location+'\\'+video_name.split('.')[0])
        base_path = self.procesed_video_location+'\\'+ video_name.split('.')[0]+'\\'
        digit = len(str(int(input.get(cv2.CAP_PROP_FRAME_COUNT))))
        fps = int(input.get(cv2.CAP_PROP_FPS))
        framecount = 0
        while True:
            _i , frame = input.read()
            if (_i):
                mediapipe_holistic = self.holistic.Holistic(static_image_mode = False, model_complexity = 2,smooth_landmarks =True, min_detection_confidence = 0.5)
                outcome = mediapipe_holistic.process(frame)
                outcome_image = np.zeros(frame.shape)
                self.drawing.draw_landmarks(outcome_image,outcome.left_hand_landmarks,self.holistic.HAND_CONNECTIONS,self.dot_spec,self.hand_spec)
                self.drawing.draw_landmarks(outcome_image,outcome.right_hand_landmarks,self.holistic.HAND_CONNECTIONS,self.dot_spec,self.hand_spec)
                self.drawing.draw_landmarks(outcome_image,outcome.face_landmarks,self.holistic.FACEMESH_TESSELATION,self.face_spec,self.face_spec)
                self.drawing.draw_landmarks(outcome_image,outcome.pose_landmarks,self.holistic.POSE_CONNECTIONS,self.pose_spec,self.pose_spec)
                mediapipe_holistic.close()
                result = outcome_image.astype(np.uint8)
                img = Image.fromarray(result)
                img.save(
                    f'{base_path}{str(framecount).zfill(digit)}.jpg',
                    optimize=True,
                    quality=10,
                )
                framecount += 1
                output.write(result)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        input.release()
        output.release()
        cv2.destroyAllWindows()
        # print("Completed key point detection for file: ", video_name)
        return 0

