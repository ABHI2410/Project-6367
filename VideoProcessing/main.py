from glob import glob
from FileConvertor import Convertor
from KeyPointDetector import KeyPointDetector

print("Start")

con = Convertor()
##con.video_clipping()
for name in sorted(glob('../Dataset/processed_videos/*.mp4')):
    con.process_video(name.split('/')[-1])

print("completed video pre processing")
