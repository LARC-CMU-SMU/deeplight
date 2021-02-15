import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import os
import sys
import random
random.seed(625742)
import time

DEBUG = 0

NO_FRAMES_MAX = 8000
#REPEAT = 20
GRID_X = 3
GRID_Y = 3
GAUSS = 0
DELTA = 5
#NUMPAT33 = 384


READ_FILE = "../multipleVideos"
EXTENSION = ".mp4"




WIDTH = 1920
HEIGHT = 1080


'''try:
    GRID_X=GRID_Y=int(sys.argv[2])
except:
    print("Using hardcoded gauss={} delta={} values".format(GAUSS,DELTA))'''

READ_FILE=sys.argv[1]
EXTENSION=sys.argv[2]
GRID_X=GRID_Y=int(sys.argv[3])



OUT_FILE = "{}-code-{}x{}".format(READ_FILE,GRID_Y,GRID_X)
READ_FILE_NAME=READ_FILE+'.'+EXTENSION
OUT_VIDEO_FILE_NAME=OUT_FILE+".mp4"
OUT_CSV_FILE_NAME=OUT_FILE+".csv"

'''if GAUSS == 1:
    prefix = "gauss_"
else:
    prefix = "none_"'''


def genPatten(code):
    patternMatrix = np.zeros((GRID_Y, GRID_X), np.float)
    for i in range(GRID_X):
        for j in range(GRID_Y):
            patternMatrix[j][i] = code & 1
            code = code >> 1
    patternMatrix = np.repeat(patternMatrix, HEIGHT/GRID_Y, axis=0)
    patternMatrix = np.repeat(patternMatrix, WIDTH/GRID_X, axis=1)
    if GAUSS == 1:
        patternMatrix = ndimage.gaussian_filter(patternMatrix, (int(WIDTH/GRID_X/16),int(HEIGHT/GRID_Y/16)), 0)
    return patternMatrix


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Initialize >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


video = cv2.VideoWriter(OUT_VIDEO_FILE_NAME,cv2.VideoWriter_fourcc('M','J','P','G'),60,(WIDTH,HEIGHT))
patfile = open(OUT_CSV_FILE_NAME, 'w')
one = np.ones((HEIGHT, WIDTH, 1), np.uint8)*255
zero = np.zeros((HEIGHT, WIDTH, 1), np.uint8)
white = cv2.merge([one*0.6, one*0.6, one*0.6]).astype(np.uint8)
red = cv2.merge([zero, zero, one])
green = cv2.merge([zero, one, zero])
blue = cv2.merge([one, zero, zero])



for col in [red,green,blue,red,green,blue,white,white,white,white]:
    video.write(col)
    patfile.write("0\n")


inputVideo = cv2.VideoCapture(READ_FILE_NAME)
frameindex = 0


for patindex in range(int(NO_FRAMES_MAX)):
    ret, frame = inputVideo.read()

    if ret:
        code=random.randint(0,(2**(GRID_X*GRID_Y))-1)


        raw = cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT)), cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(raw)
        lnorm = np.square(l*100.0/255-50)
        upper = lnorm*0.015
        lower = np.sqrt(lnorm + 20)
        delt = np.divide((upper + lower), lower)*DELTA*2.55
        mask = genPatten(code)
        lmod = l + np.multiply(mask, delt)
        calib = np.max(lmod)
        if calib > 255:
            lmod = np.uint8((lmod/calib)*255)
        else:
            lmod = np.uint8(lmod)
        img = cv2.merge([lmod, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        video.write(img)
        patfile.write(str(code) + "\n")
        if patindex%20==0: print("Pattern: " + str(patindex) + ", Image name: " + str(frameindex))
    else:
        break
    frameindex += 1
inputVideo.release()



for col in [red,green,blue,red,green,blue,white]:
    video.write(col)
    patfile.write("0\n")

patfile.close()
video.release()
cv2.destroyAllWindows()
