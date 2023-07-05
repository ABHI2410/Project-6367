from glob import glob
from FileConvertor import Convertor
from KeyPointDetector import KeyPointDetector
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Process
print("Start")

def process(file):
    con = Convertor()
    con.process_video(file)
# con.video_clipping()
files = []
for counter,name in enumerate(sorted(glob('../clip_videos/*.mp4'))):
    files.append(name.split('/')[-1])
    # con.process_video(files[counter])
    # print(counter)
# con = Convertor()
# con.process_video(files[0])
print(len(files))
with concurrent.futures.ProcessPoolExecutor (max_workers=1) as executor:
    executor.map(process,files)


print("completed video pre processing")
