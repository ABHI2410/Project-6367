import cv2
import mediapipe as mp
from multiprocessing import Pool
from glob import glob

def process_video(file):
    

# Define the function to process a single frame
def process_frame(frame):
    # Initialize the mediapipe solution
    mp_holistic = mp.solutions.holistic.Holistic()

    # Process the frame
    with mp_holistic.process(frame) as results:
        # Access the landmarks
        landmarks = results.pose_landmarks

        # Do further processing with the landmarks
        # ...

    # Release resources
    mp_holistic.close()

    # Return the processed result
    return processed_result


if __name__ == '__main__':
    # Define the video file paths
    video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4', ...]
    files = []
    for counter,name in enumerate(sorted(glob('../clip_videos/*.mp4'))):
        files.append(name.split('/')[-1])
    # Define the number of processes to use
    num_processes = 4

    # Create a process pool
    pool = Pool(num_processes)

    # Process the videos in parallel
    processed_results = pool.map(process_video, files)

    # Close the process pool
    pool.close()
    pool.join()

    # Process the results
    for result in processed_results:
        # Do something with the processed results
        # ...
        pass
