import cv2 
import mediapipe as mp
import numpy as np
from glob import glob
import os


# Colors 
WHITE = (225, 225, 225)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class KeyPointDetector:
    def __init__(self) -> None:
        self.holistic = mp.solutions.holistic
        self.mediapipe_holistic = self.holistic.Holistic(static_image_mode = True, min_detection_confidence = 0.6)
        self.drawing = mp.solutions.drawing_utils
        self.face_spec = self.drawing.DrawingSpec(color=WHITE, thickness=1, circle_radius=1)
        self.pose_spec = self.drawing.DrawingSpec(color=WHITE, thickness=3, circle_radius=3)
        self.hand_spec = self.drawing.DrawingSpec(color=WHITE, thickness=3, circle_radius=3)
        self.dot_spec = self.drawing.DrawingSpec(color=RED, thickness=2, circle_radius=3)
        self.image_folder = "../Dataset/images"
        self.frames_folder = "../Dataset/frames"

    def keypoints(self):
        files = []
        for name in sorted(glob('../Dataset/frames/*.jpg')):
            files.append(name)
        images  = {name: cv2.imread(name)[:,:,::-1] for name in files}

        for name,image in images.items():

            outcome = self.mediapipe_holistic.process(image)
            outcome_image = image.copy()

            self.drawing.draw_landmarks(outcome_image,outcome.left_hand_landmarks,self.holistic.HAND_CONNECTIONS,self.dot_spec,self.hand_spec)
            self.drawing.draw_landmarks(outcome_image,outcome.right_hand_landmakrs,self.holistic.HAND_CONNECTIONS,self.dot_spec,self.hand_spec)
            self.drawing.draw_landmarks(outcome_image,outcome.face_landmarks,self.holistic.FACEMESH_TESSELATION,self.face_spec,self.face_spec)
            self.drawing.draw_landmarks(outcome_image,outcome.pose_landmarks,self.holistic.POSE_CONNECTIONS,self.pose_spec,self.pose_spec)

            save_name = self.image_folder+os.path.basename(name)
            cv2.imwrite(save_name,outcome_image)


