import cv2
import os
import shutil


def create_movie(frames_path, framereate, video_name, del_frames):
    print("Creating movie...", end='')

    # Create a video from the images
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Get the list of image filenames in the directory
    img_names = os.listdir(frames_path)
    img_names.sort()

    framesize = cv2.imread(os.path.join(frames_path, img_names[0])).shape
    framesize = (framesize[1], framesize[0])

    # Create the video writer object
    video = cv2.VideoWriter(video_name, fourcc, framereate, framesize)

    # Loop through the image filenames and add each frame to the video
    for img_name in img_names:
        img_path = os.path.join(frames_path, img_name)
        img = cv2.imread(img_path)
        video.write(img)

    # Release the video writer object and close the video file
    video.release()
    cv2.destroyAllWindows()

    if del_frames:
        shutil.rmtree(frames_path)

    print(" Done.")
