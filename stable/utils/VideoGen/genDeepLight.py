#To generate plain data with RSCode, with landmark, 5-bit sequence, RS 50%
import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import os
import sys
import random
import time
sys.path.append("./RSCode")
from codec import *

WIDTH = 1920
HEIGHT = 1080
GRID_X = 10
GRID_Y = 10
DELTA = [3,2,2,2]

sym_bits = 5 #Number of bit per symbol
block_len = 20 #Number of symbols in a block (we transmit data in a block)
percentage = 50
ratio = percentage / 100.0 #Ratio of data symbol in a block. The rest are error correction code
msg_len = int(ratio*block_len) #Number of data symbols
codec = Codec(sym_bits=sym_bits, block_len=block_len, code_ratio=ratio)
datalen = int(percentage*GRID_X*GRID_Y/100)

def embedManchester(data, width, height, img): #Generate a mask. data: 2D numpy array; width, height: size of the image
    mask = np.repeat(data.astype(np.float), height/data.shape[0], axis=0)
    mask = np.repeat(mask, width/data.shape[1], axis=1)
    mask = mask * 2 - 1 # change to +1.0 and -1.0
    b,g,r = cv2.split(img)
    level = np.zeros((GRID_Y, GRID_X), dtype=np.float)
    cell_width = int(WIDTH/GRID_X)
    cell_height = int(HEIGHT/GRID_Y)
    for i in range(GRID_X):
        for j in range(GRID_Y):
            tem = np.mean(b[j*cell_height:(j+1)*cell_height, i*cell_width:(i+1)*cell_width])
            if (tem < 180):
                if (tem < 75):#75
                    if (tem < 25): #35
                        level[j,i] = DELTA[0] #6
                    else:
                        level[j,i] = DELTA[1] #5
                else:
                    level[j,i] = DELTA[2] #4
            else:
                level[j,i] = DELTA[3] #4
    level = np.repeat(level, HEIGHT/GRID_Y, axis=0)
    level = np.repeat(level, WIDTH/GRID_X, axis=1)
    bp = b.astype(np.float) + mask*level
    bmin = np.min(bp)
    if bmin < 0:
        bp = bp - bmin
    bmax = np.max(bp)
    if bmax > 255:
        bp = bp * 255.0 / bmax
    bn = b.astype(np.float) - mask*level
    bmin = np.min(bn)
    if bmin < 0:
        bn = bn - bmin
    bmax = np.max(bn)
    if bmax > 255:
        bn = bn * 255.0 / bmax
    frame0 = cv2.merge([bp.astype(np.uint8), g, r])
    frame1 = cv2.merge([bn.astype(np.uint8), g, r])
    return frame0, frame1



patterns = np.load(sys.argv[1])
pattern_index = 0
seq_num = 0
videoin = cv2.VideoCapture(sys.argv[2])
frame_count = int(videoin.get(cv2.CAP_PROP_FRAME_COUNT))
print("Frame count: " + str(frame_count))
if not((".AVI" in sys.argv[3]) or (".avi" in sys.argv[3])):
    print("Output file should be .AVI to avoid inter-frame compression")
    exit(0)
videoout = cv2.VideoWriter(sys.argv[3],cv2.VideoWriter_fourcc('M','J','P','G'),60,(WIDTH,HEIGHT))
num_frame = int(sys.argv[4])
frame_index = 0
for index in range(num_frame):
    ret,frame = videoin.read()
    if ret:
        raw_data = patterns[pattern_index, 0:datalen]
        tem_seq = seq_num
        for bit in range(datalen-10, datalen-5):
            raw_data[bit] = tem_seq & 0x01
            tem_seq = tem_seq >> 1
        csum = 0
        for sym in range(int(datalen/sym_bits)-1):
            value = 0
            for bit in range(sym*sym_bits, sym*sym_bits+sym_bits):
                value = value << 1
                value += raw_data[bit]
            csum = ((csum & 0x1F) >> 1) + ((csum & 0x1) << 4)
            csum = (csum + value) & 0x1F
        for bit in range(datalen-5, datalen):
            raw_data[bit] = csum & 0x01
            csum = csum >> 1
        rs_data = codec.encode(raw_data)
        rs_reshape = np.reshape(rs_data, (GRID_Y, GRID_X))
        frame0, frame1 = embedManchester(rs_reshape, WIDTH, HEIGHT, frame)
        videoout.write(frame0)
        videoout.write(frame1)
        pattern_index += 1
        frame_index += 1
        seq_num = seq_num + 1
videoout.release()
videoin.release()
cv2.destroyAllWindows()