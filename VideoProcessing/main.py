from glob import glob
from FileConvertor import Convertor
from tqdm.contrib.concurrent import process_map



def process(file):
    con = Convertor()
    con.video_2_frame(file)



print("Start")
# con = Convertor()
# con.video_clipping()
files = []
for counter,name in enumerate(sorted(glob('../clip_videos/*.mp4'))):
    files.append(name.split('/')[-1])
process_map(process,files,max_workers=4, chunksize=1)
print("completed video pre processing")

