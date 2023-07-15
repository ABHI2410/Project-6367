from glob import glob
from FileConvertor import Convertor
from KeyPointDetector import KeyPointDetector
from tqdm import tqdm
from time import sleep
from multiprocessing import Process, Pool
import pandas as pd
import os 

# def process(file):
#     con = Convertor()
#     con.process_video(file)



# print("Start")
con = Convertor()
# con.video_clipping()

# files = []
# for counter,name in enumerate(sorted(glob('/mnt/d/Project-6367/Dataset/processed_videos/*.mp4'))):
#     files.append(name.split('/')[-1])
# con.identify_human(files[0])
con.pre_processing()
# process(files[0])
# # with Pool(processes=4) as pool:
# #     tqdm(pool.map(process,files))
# print("completed video pre processing")
# #3100, 7/13/2023,12:23




