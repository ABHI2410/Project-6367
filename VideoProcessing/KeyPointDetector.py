import mediapipe as mp
import numpy as np


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
    hand_spec = drawing.DrawingSpec(color=GREEN, thickness=3, circle_radius=3)
    dot_spec = drawing.DrawingSpec(color=RED, thickness=2, circle_radius=3)


    def keypoints(self,image):
        mediapipe_holistic = self.holistic.Holistic(static_image_mode = False, model_complexity = 2,smooth_landmarks =True, min_detection_confidence = 0.5)
        outcome = mediapipe_holistic.process(image)
        outcome_image = np.zeros(image.shape)
        self.drawing.draw_landmarks(outcome_image,outcome.left_hand_landmarks,self.holistic.HAND_CONNECTIONS,self.dot_spec,self.hand_spec)
        self.drawing.draw_landmarks(outcome_image,outcome.right_hand_landmarks,self.holistic.HAND_CONNECTIONS,self.dot_spec,self.hand_spec)
        self.drawing.draw_landmarks(outcome_image,outcome.face_landmarks,self.holistic.FACEMESH_TESSELATION,self.face_spec,self.face_spec)
        self.drawing.draw_landmarks(outcome_image,outcome.pose_landmarks,self.holistic.POSE_CONNECTIONS,self.pose_spec,self.pose_spec)
        mediapipe_holistic.close()
        return outcome_image
    