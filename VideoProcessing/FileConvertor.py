import cv2
import os
import pandas as pd
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from KeyPointDetector import KeyPointDetector
from multiprocessing import Pool
import h5py
from tqdm.contrib.concurrent import process_map
import numpy as np
import mediapipe as mp
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Convertor:
    def __init__(self) -> None:
        self.progress_file = "progress.txt"
        self.output_folder = "ProcessData/"
        directory = os.getcwd()
        os.chdir("../../../../../d")
        os.chdir("Project-6367/Dataset/raw_videos")
        self.video_location = os.getcwd()
        os.chdir("../raw_data/")
        self.csv_location = os.getcwd() + '/' + "how2sign_realigned_train.csv"
        os.chdir("../processed_videos")
        self.video_clip_location = os.getcwd()
        os.chdir("../../../../c/Users/patel/Project-6367/VideoProcessing")     
    












    def video_clipping(self):

        df = pd.read_csv(self.csv_location, sep= "	", header= 0)
        def clip(row):

            video_name = row.VIDEO_NAME
            start_time = row.START_REALIGNED
            end_time = row.END_REALIGNED
            sentence_name = row.SENTENCE_NAME

            ffmpeg_extract_subclip(self.video_location +'/'+ video_name + ".mp4", start_time, end_time, targetname=f"{self.procesed_video_location}/{sentence_name}.mp4")
        # with Pool(processes=4) as pool:
        # process_map(clip,df.itertuples(index=True, name='Pandas'))
        for ites in df.itertuples(index=True, name='Pandas'):
            clip(ites)




















    def identify_human(self,video_name):

        base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                            score_threshold=0.5,
                                            max_results = 1)
        detector = vision.ObjectDetector.create_from_options(options)
        obj = KeyPointDetector()
        video = cv2.VideoCapture(self.video_clip_location+"/"+video_name+".mp4")
        origin_x = 0
        origin_y = 0
        width = 0
        height = 0

        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            if origin_x == 0 and origin_y == 0 and width == 0 and height == 0:
                detection_result = detector.detect(mp_image)
                if detection_result.detections:
                    category_detected = detection_result.detections[0].categories[0].category_name
                    if category_detected == "person":

                        origin_x = detection_result.detections[0].bounding_box.origin_x
                        origin_y = detection_result.detections[0].bounding_box.origin_y
                        width = detection_result.detections[0].bounding_box.width
                        height = detection_result.detections[0].bounding_box.height
                        break
        detector.close()
        video.release()
        cv2.destroyAllWindows()

        return [origin_y, origin_y+height, origin_x, origin_x+width]
    























    def detect_hand_landmarks(self,frame):
        result = {}
        mp_hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.6)
        results = mp_hands.process(frame)
        try:
            res_at_0_position = results.multi_handedness[0].classification[0].label
        except TypeError:
            mp_hands.close()
            return result
        result[res_at_0_position] = results.multi_hand_landmarks[0]
        try:
            res_at_1_position = results.multi_handedness[1].classification[0].label
        except IndexError:
            mp_hands.close()
            return result      
        result[res_at_1_position] = results.multi_hand_landmarks[1]  
        mp_hands.close()
        return result

    def detect_pose_landmarks(self,frame):
        mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.6)
        results = mp_pose.process(frame)
        mp_pose.close()
        return results

    def detect_face_landmarks(self,frame):
        mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.6)
        results = mp_face.process(frame)
        mp_face.close()
        return results




















    def process_video(self,video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(self.output_folder, video_name + ".h5")
        previous_frame_landmarks = {
            'Right' : np.empty((0, 3)),
            'Left' : np.empty((0, 3)),
        }
        previous_frame_pose = np.empty((0, 3))
        previous_frame_face = np.empty((0, 3))
        # Check if the video has already been processed
        if video_name in self.load_completed_videos():
            print(f"Skipping {video_name}. Video already processed.")
            return

        # Open video file for reading
        video = cv2.VideoCapture(video_path)

        # Create HDF5 file for storing landmarks and attributes
        hdf5_file = h5py.File(output_file, "w")

        # Find the corresponding row in the CSV file
        df = pd.read_csv(self.csv_location, sep='\t')
        row = df.loc[df['SENTENCE_NAME'] == video_name]

        if not row.empty:
            # Retrieve the english_text value from the CSV row
            english_text = row['SENTENCE'].iloc[0]
        else:
            english_text = "Unknown"  # Set a default value if not found

        # Create a group for the video
        group = hdf5_file.create_group(video_name)

        # Get total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


        # Iterate over each frame in the video
        for frame_index in range(total_frames):
            ret, frame = video.read()
            if not ret:
                break
            
            # Detech Humans
            human_location = self.identify_human(video_name)
            cropped_image = frame[human_location[0]:human_location[1], human_location[2]:human_location[3]]
            cv2.imwrite("temp.jpg", cropped_image)
            # Detect hand landmarks
            hand_landmarks = self.detect_hand_landmarks(cropped_image)

            # Detect pose landmarks
            pose_landmarks = self.detect_pose_landmarks(cropped_image)

            # Detect face landmarks
            face_landmarks = self.detect_face_landmarks(cropped_image)

            # Store the hand landmarks as a dataset within the group
            if len(hand_landmarks) == 2:
                for key,val in hand_landmarks.items():
                    hand_landmarks_data = np.array([[lm.x, lm.y, lm.z] for lm in val.landmark])
                    previous_frame_landmarks[key] = hand_landmarks_data
                    group.create_dataset(f'hand_landmarks_{key}_{frame_index}', data=hand_landmarks_data)
            elif len(hand_landmarks) == 1:
                key = list(hand_landmarks.keys())[0]
                hand_landmarks_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks[key].landmark])
                group.create_dataset(f'hand_landmarks_{key}_{frame_index}', data=hand_landmarks_data)
                if key == "Right":
                    #get previous frames left hand landmarks
                    group.create_dataset(f'hand_landmarks_Left_{frame_index}', data=previous_frame_landmarks["Left"])
                else:
                    #get previous frames right hand landmarks
                    group.create_dataset(f'hand_landmarks_Right_{frame_index}', data=previous_frame_landmarks["Right"])
            else:
                # If no hand landmarks were detected, store an empty array
                group.create_dataset(f'hand_landmarks_Right_{frame_index}', data=previous_frame_landmarks["Right"])
                group.create_dataset(f'hand_landmarks_Left_{frame_index}', data=previous_frame_landmarks["Left"])

            # Store the pose landmarks as a dataset within the group
            if pose_landmarks is not None and pose_landmarks.pose_landmarks:
                pose_landmarks_data = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.pose_landmarks.landmark])
                previous_frame_pose = pose_landmarks_data
                group.create_dataset(f'pose_landmarks_{frame_index}', data=pose_landmarks_data)
            else:
                # If no pose landmarks were detected, store an empty array
                group.create_dataset(f'pose_landmarks_{frame_index}', data=previous_frame_pose)

            # Store the face landmarks as a dataset within the group
            if face_landmarks is not None and face_landmarks.multi_face_landmarks:
                face_landmarks_data = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.multi_face_landmarks[0].landmark])
                previous_frame_face = face_landmarks_data
                group.create_dataset(f'face_landmarks_{frame_index}', data=face_landmarks_data)
            else:
                # If no face landmarks were detected, store an empty array
                group.create_dataset(f'face_landmarks_{frame_index}', data=previous_frame_face)

            # Store the English text as an attribute of the group
            group.attrs['english_text'] = english_text

        # Release the video capture
        video.release()

        # Close the HDF5 file
        hdf5_file.close()

        # Append the completed video to the progress file
        with open(self.progress_file, "a") as f:
            f.write(f"{video_name}\n")











    def load_completed_videos(self):
        if not os.path.exists(self.progress_file):
            return []
        
        with open(self.progress_file, "r") as f:
            return [line.strip() for line in f.readlines()]
        
















    def pre_processing(self):
        video_folder =  self.video_clip_location
        csv_file = self.csv_location
        progress_file = self.progress_file

        # Read the CSV file
        df = pd.read_csv(csv_file, sep='\t')

        # Get the list of video files
        video_files = [os.path.join(video_folder, filename) for filename in os.listdir(video_folder) if filename.endswith(".mp4")]

        # Load the list of completed videos
        completed_videos = self.load_completed_videos()

        # Create a Pool of worker processes
        num_processes = 4  # Adjust the number of processes as per your system capabilities
        pool = Pool(processes=num_processes)

        try:
            # Process the videos in parallel with a progress bar
            with tqdm(total=len(video_files), desc="Videos processed") as pbar:
                for _ in pool.imap_unordered(self.process_video, video_files):
                    pbar.update()

        except KeyboardInterrupt:
            # If the code is interrupted with a keyboard interrupt (Ctrl+C)
            # Save the progress so far
            with open(progress_file, "w") as f:
                for video in completed_videos:
                    f.write(f"{video}\n")
            print("\nProcessing interrupted. Progress saved.")

        # Close the worker processes
        pool.close()
        pool.join()