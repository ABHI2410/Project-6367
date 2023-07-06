from glob import glob
from FileConvertor import Convertor
from KeyPointDetector import KeyPointDetector
from tqdm import tqdm
from time import sleep
from multiprocessing import Process, Pool


def process(count, file):
    con = Convertor()
    con.process_video(file)



print("Start")
# con = Convertor()
# con.video_clipping()
files = []
for counter,name in enumerate(sorted(glob('../clip_videos/*.mp4'))):
    files.append(name.split('/')[-1])
with Pool(processes=4 , maxtasksperchild= 2) as pool:
    tqdm(pool.map(process,files))
print("completed video pre processing")
