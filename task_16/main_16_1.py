import numpy as np
import cv2

video_to_process = 'street_camera.mp4'

# create VideoCapture object for further video processing
captured_video = cv2.VideoCapture(video_to_process)
#captured_video = cv2.VideoCapture(0)
# check video capture status
if not captured_video.isOpened:
    print("Unable to open: " + video_to_process)
    exit(0)

# instantiate background subtraction
#bs = cv2.bgsegm.createBackgroundSubtractorGSOC()
#bs = cv2.bgsegm.createBackgroundSubtractorCNT()
#bs = cv2.createBackgroundSubtractorMOG2()
bs = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = captured_video.read()
    if frame is None:
        break

    fgMask = bs.apply(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)



    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break