from glob import glob
from FileConvertor import Convertor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool


def process(file):
    con = Convertor()
    con.video_2_frame(file)


if __name__ == '__main__':
    print("Start")
    # con = Convertor()
    # con.video_clipping()
    files = []
    for counter,name in enumerate(sorted(glob("D:\\Project-6367\\Dataset\\clip_videos\\*.mp4"))):
        files.append(name.split('\\')[-1])
    # process(files[0])
    process_map(process,files,chunksize=1, max_workers = 3)
    # print(len(files))
    # with Pool(processes=4) as pool:
    #     pool.map(process,files)

    print("completed video pre processing")

