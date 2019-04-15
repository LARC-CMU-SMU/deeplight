import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import os
import sys
import random
import time

WIDTH = 1920
HEIGHT = 1080
NUMPATTERN = 20000
REPEAT = 20
GRID_X = 40
GRID_Y = 40
GAUSS = 0
DELTA = 5
NUMPAT33 = 384
FILE="butterfly"
DEBUG = 0


try:
    FILE=sys.argv[1]
    GAUSS = int(sys.argv[2])
    DELTA = int(sys.argv[3])
except:
	print("Using hardcoded gauss={} delta={} values".format(GAUSS,DELTA))

EXTENSION=".mp4"
FILE_NAME=FILE+EXTENSION

if GAUSS == 1:
    prefix = "gauss_"
else:
    prefix = "none_"


def genPatten(index):
    info = np.zeros((GRID_Y, GRID_X), np.float)
    tem = patterns[index]
    for i in range(GRID_X):
        for j in range(GRID_Y):
            if tem & 0x01 > 0:
                info[j][i] = 0
            else:
                info[j][i] = 1
            tem = tem >> 1
    info = np.repeat(info, HEIGHT/GRID_Y, axis=0)
    info = np.repeat(info, WIDTH/GRID_X, axis=1)
    if GAUSS == 1:
        info = ndimage.gaussian_filter(info, (int(WIDTH/GRID_X/16),int(HEIGHT/GRID_Y/16)), 0)
    return info

#Generate patterns
patterns = np.zeros(NUMPATTERN, dtype=np.int)
codebook = np.zeros(NUMPAT33, dtype=np.int)
pattern33 = np.zeros(NUMPATTERN, dtype=np.int)

tem = np.arange(512)
random.shuffle(tem)
for i in range(NUMPAT33):
    codebook[i] = tem[i]

for i in range(NUMPATTERN):
    value = patterns[i]
    while value in patterns[0:i+1]:
        value = random.randint(0, 2**(GRID_X * GRID_Y))
    patterns[i] = value
print("Finish preprocessing")



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Initialize >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
video = cv2.VideoWriter(FILE+prefix + str(NUMPAT33) + "_" + str(DELTA) + "_mix.mp4",cv2.VideoWriter_fourcc('M','J','P','G'),60,(WIDTH,HEIGHT))
patfile = open(FILE+prefix + str(NUMPAT33) + "_" + str(DELTA) + "_mix.csv", 'w')
one = np.ones((HEIGHT, WIDTH, 1), np.uint8)*255
zero = np.zeros((HEIGHT, WIDTH, 1), np.uint8)
white = cv2.merge([one*0.6, one*0.6, one*0.6]).astype(np.uint8)
red = cv2.merge([zero, zero, one])
green = cv2.merge([zero, one, zero])
blue = cv2.merge([one, zero, zero])
video.write(red)
patfile.write("0\n")
video.write(green)
patfile.write("0\n")
video.write(blue)
patfile.write("0\n")
video.write(red)
patfile.write("0\n")
video.write(green)
patfile.write("0\n")
video.write(blue)
patfile.write("0\n")






for i in range(4):
    video.write(white)
    patfile.write("0\n")
inputVideo = cv2.VideoCapture(FILE_NAME)
frameindex = 0
# for i in range(200):
#     ret, frame = inputVideo.read()
#     frameindex += 1
for patindex in range(int(NUMPATTERN)):
    ret, frame = inputVideo.read()
    if ret:
        raw = cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT)), cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(raw)
        lnorm = np.square(l*100.0/255-50)
        upper = lnorm*0.015
        lower = np.sqrt(lnorm + 20)
        delt = np.divide((upper + lower), lower)*DELTA*2.55
        mask = genPatten(patindex);
        lmod = l + np.multiply(mask, delt)
        calib = np.max(lmod)
        if calib > 255:
            lmod = np.uint8((lmod/calib)*255)
        else:
            lmod = np.uint8(lmod)
        img = cv2.merge([lmod, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        video.write(img)
        patfile.write(str(patterns[patindex]) + "\n")
        print("Pattern: " + str(patindex) + ", Image name: " + str(frameindex))
    else:
        break
    frameindex += 1
inputVideo.release()
video.write(red)
patfile.write("0,0,none\n")
video.write(green)
patfile.write("0,0,none\n")
video.write(blue)
patfile.write("0,0,none\n")
video.write(red)
patfile.write("0,0,none\n")
video.write(green)
patfile.write("0,0,none\n")
video.write(blue)
patfile.write("0,0,none\n")

patfile.close()
video.release()
cv2.destroyAllWindows()