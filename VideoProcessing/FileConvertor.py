import cv2
import os


class Convertor:
    def __init__(self) -> None:
        self.video_location = "../Dataset/raw_videos"
        self.frame_location = "../Dataset/frames"
        self.image_location = "../Dataset/images"

    def video2frame(self,videoname):
        def video_2_images(location,framelocation,image):
            i = 0
            interval = 1
            limiter = 1800
            cap = cv2.VideoCapture(location)
            fps = cap.get(cv2.CAP_PROP_FPS)
            while(cap.isOpened()):
                flag, frame = cap.read()  
                if flag == False:  
                        break
                if i == limiter*interval:
                        break
                if i % interval == 0:    
                    cv2.imwrite(framelocation+image % str(int(i/interval)).zfill(6), frame)
                i += 1 
            cap.release()
            return fps, i, interval
        video = videoname
        image_file = "%s.jpg"
        self.fps,self.i,self.interval = video_2_images(self.video_location+video, self.frame_location,image_file)

    def images2video(self,):
        with_sound = True #@param {type:"boolean"}
        fps_r = fps/interval

        print('making movie...')
        if with_sound ==  True:  
            ! ffmpeg -y -r $fps_r -i images/%6d.jpg -vcodec libx264 -pix_fmt yuv420p -loglevel error out.mp4
            # audio extraction/addition
            print('preparation for sound...')
            ! ffmpeg -y -i $video_file -loglevel error sound.mp3
            ! ffmpeg -y -i out.mp4 -i sound.mp3 -loglevel error output.mp4
        else:
            ! ffmpeg -y -r $fps_r -i images/%6d.jpg -vcodec libx264 -pix_fmt yuv420p -loglevel error output.mp4

        display_mp4('output.mp4')


