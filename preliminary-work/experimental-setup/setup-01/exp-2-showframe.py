import numpy as np
import cv2
import sys
import time

WIDTH = 1920
HEIGHT = 1080

video = cv2.VideoCapture('naturenone_384_5_mix.mp4')
pattern = open('naturenone_384_5_mix.csv')
label = open('nature_16090_5_split.csv', 'w')
label.write("0,0\n")
black = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow('frame',black)
cv2.waitKey()
while(video.isOpened()):
    ret, frame = video.read()
    pat = pattern.readline()
    if ret:
        label.write(str(time.clock_gettime(time.CLOCK_MONOTONIC)) + "," + pat)
        cv2.imshow('frame',frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()
pattern.close()
label.close()