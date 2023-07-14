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
        print("Arrived")
        df = pd.read_csv(self.csv_location, sep= "	", header= 0)
        def clip(row):
            print("Called Clip")
            print()
            video_name = row.VIDEO_NAME
            start_time = row.START_REALIGNED
            end_time = row.END_REALIGNED
            sentence_name = row.SENTENCE_NAME
            print(video_name,start_time,end_time,sentence_name)
            ffmpeg_extract_subclip(self.video_location +'/'+ video_name + ".mp4", start_time, end_time, targetname=f"{self.procesed_video_location}/{sentence_name}.mp4")
        # with Pool(processes=4) as pool:
        # process_map(clip,df.itertuples(index=True, name='Pandas'))
        for ites in df.itertuples(index=True, name='Pandas'):
            clip(ites)

    def process_video_deprycated(self,video_name):
        if os.path.exists(self.procesed_video_location + '/' + video_name):
            print("Video Processing completed already")
            return 0
        input = cv2.VideoCapture(self.video_clip_location + '/' + video_name)
        output = cv2.VideoWriter(
            self.procesed_video_location + '/' + video_name,
            cv2.VideoWriter_fourcc('m','p','4','v'),
            30, 
            (1280,720)
            )
        video_frames_outcomes = []
        obj = KeyPointDetector()
        while input.isOpened():
            _i , frame = input.read()
            if (_i):
                frame_output = obj.keypoints(frame)     
                video_frames_outcomes.append(frame_output)
                output.write(frame_output)

                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                break
        del obj
        input.release()
        cv2.destroyAllWindows()
        df = pd.read_csv(self.csv_location, sep= "	", header= 0)

        return 0
    

    def detect_hand_landmarks(self,frame):
        mp_hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.6)
        results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_hands.close()
        return results

    def detect_pose_landmarks(self,frame):
        mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.6)
        results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_pose.close()
        return results

    def detect_face_landmarks(self,frame):
        mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.6)
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_face.close()
        return results

    def process_video(self,video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(self.output_folder, video_name + ".h5")
        
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

            # Detect hand landmarks
            hand_landmarks = self.detect_hand_landmarks(frame)

            # Detect pose landmarks
            pose_landmarks = self.detect_pose_landmarks(frame)

            # Detect face landmarks
            face_landmarks = self.detect_face_landmarks(frame)

            # Store the hand landmarks as a dataset within the group
            if hand_landmarks is not None and hand_landmarks.multi_hand_landmarks:
                hand_landmarks_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.multi_hand_landmarks[0].landmark])
                group.create_dataset(f'hand_landmarks_{frame_index}', data=hand_landmarks_data)
            else:
                # If no hand landmarks were detected, store an empty array
                hand_landmarks_data = np.empty((0, 3))
                group.create_dataset(f'hand_landmarks_{frame_index}', data=hand_landmarks_data)

            # Store the pose landmarks as a dataset within the group
            if pose_landmarks is not None and pose_landmarks.pose_landmarks:
                pose_landmarks_data = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.pose_landmarks.landmark])
                group.create_dataset(f'pose_landmarks_{frame_index}', data=pose_landmarks_data)
            else:
                # If no pose landmarks were detected, store an empty array
                pose_landmarks_data = np.empty((0, 3))
                group.create_dataset(f'pose_landmarks_{frame_index}', data=pose_landmarks_data)

            # Store the face landmarks as a dataset within the group
            if face_landmarks is not None and face_landmarks.multi_face_landmarks:
                face_landmarks_data = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.multi_face_landmarks[0].landmark])
                group.create_dataset(f'face_landmarks_{frame_index}', data=face_landmarks_data)
            else:
                # If no face landmarks were detected, store an empty array
                face_landmarks_data = np.empty((0, 3))
                group.create_dataset(f'face_landmarks_{frame_index}', data=face_landmarks_data)

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