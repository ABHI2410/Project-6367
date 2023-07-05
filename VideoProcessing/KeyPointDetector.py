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
    holistic = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils
    face_spec = drawing.DrawingSpec(color=BLUE, thickness=1, circle_radius=1)
    pose_spec = drawing.DrawingSpec(color=WHITE, thickness=3, circle_radius=3)
    hand_spec = drawing.DrawingSpec(color=BLACK, thickness=3, circle_radius=3)
    dot_spec = drawing.DrawingSpec(color=RED, thickness=2, circle_radius=3)
    image_folder = "../Dataset/images"
    frames_folder = "../Dataset/frames"

    def keypoints(self,image):
        mediapipe_holistic = self.holistic.Holistic(static_image_mode = True, min_detection_confidence = 0.6)
        outcome = mediapipe_holistic.process(image)
        # outcome_image = image.copy()
        self.drawing.draw_landmarks(image,outcome.left_hand_landmarks,self.holistic.HAND_CONNECTIONS,self.dot_spec,self.hand_spec)
        self.drawing.draw_landmarks(image,outcome.right_hand_landmarks,self.holistic.HAND_CONNECTIONS,self.dot_spec,self.hand_spec)
        self.drawing.draw_landmarks(image,outcome.face_landmarks,self.holistic.FACEMESH_TESSELATION,self.face_spec,self.face_spec)
        self.drawing.draw_landmarks(image,outcome.pose_landmarks,self.holistic.POSE_CONNECTIONS,self.pose_spec,self.pose_spec)
        mediapipe_holistic.close()
        return image
    