import time
import numpy as np
from numpy.random import seed
seed(625742)
import argparse
import cv2
import sys
import os
import struct
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import importlib
from keras.models import Model
sys.path.append("./RSCode")
from codec import *
sys.path.append("./models")
lightnet = importlib.import_module("LightNet")
screennet = importlib.import_module("ScreenNet")

def fitting(points, dist, direction, size, step):
    #A RANSAC inspired (Without ramdom sample as the contour points are sequential) fitting algorithm to find the border. 
    max_fit = -1
    max_arg = None
    head_index = 0
    while head_index < points.shape[0]: 
        if head_index + size > points.shape[0]:
            tem_points = np.vstack([points[head_index:],points[:head_index + size - points.shape[0]]])
        else:
            tem_points = points[head_index:head_index + size]
        vx, vy, cx, cy = cv2.fitLine(tem_points, cv2.DIST_L2, 0, 0.01, 0.01)
        a = vy
        b = -vx
        c = vx*cy-vy*cx
        head_index += step
        if direction == "V":
            if (b != 0) and abs(a/b) <= 1.0:
                continue
        elif direction == "H":
            if (a != 0) and abs(b/a) <= 1.0:
                continue
        else:
            print("Direction not supported: " + direction)
            return None

        inliners = points[np.where(np.abs(a*points[:,0] + b*points[:,1] + c)/np.sqrt(a**2 + b**2) < dist)]

        if inliners.shape[0] > max_fit:
            max_fit = inliners.shape[0]
            vx, vy, cx, cy = cv2.fitLine(inliners, cv2.DIST_L2, 0, 0.01, 0.01)
            a = vy
            b = -vx
            c = vx*cy-vy*cx
            max_arg = (a, b, c)
    return max_arg

def intersect(a1, b1, c1, a2, b2, c2):
    x = (c2*b1-c1*b2)/(a1*b2-a2*b1)
    y = (c2*a1-c1*a2)/(b1*a2-b2*a1)
    return (int(x), int(y))

def findCorners(mask, dist, size, step):
    #Estimate screen edges (4 comers: lefttop, rightop, rightbottom, leftbottom). 
    ret, thresh = cv2.threshold(mask, 180, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours, key=cv2.contourArea)
    bestContour = contours[-1]
    M = cv2.moments(bestContour)
    xMid = int(M["m10"] / M["m00"])
    yMid = int(M["m01"] / M["m00"])

    points = []
    for i in range(bestContour.shape[0]):
        if bestContour[i-1,0,0] == bestContour[i,0,0]: # same X
            y = bestContour[i-1,0,1]
            dy = 1
            if bestContour[i-1,0,1] > bestContour[i,0,1]:
                dy = -1
            while y != bestContour[i,0,1]:
                points.append((bestContour[i,0,0], y))
                y += dy
        elif bestContour[i-1,0,1] == bestContour[i,0,1]: # same Y
            x = bestContour[i-1,0,0]
            dx = 1
            if bestContour[i-1,0,0] > bestContour[i,0,0]:
                dx = -1
            while x != bestContour[i,0,0]:
                points.append((x, bestContour[i,0,1]))
                x += dx
        elif abs(bestContour[i-1,0,0] - bestContour[i,0,0]) == abs(bestContour[i-1,0,1] - bestContour[i,0,1]):
            x = bestContour[i-1,0,0]
            y = bestContour[i-1,0,1]
            dx = 1
            dy = 1
            if bestContour[i-1,0,0] > bestContour[i,0,0]:
                dx = -1
            if bestContour[i-1,0,1] > bestContour[i,0,1]:
                dy = -1
            while x != bestContour[i,0,0]:
                points.append((x, y))
                x += dx
                y += dy
        else:
            print("============== SHOULDN'T OCCUR ================")

    left_points = []
    right_points = []
    top_points = []
    bottom_points = []
    for i in range(len(points)):
        if points[i][0] <= xMid:
            left_points.append(points[i])
        if points[i][0] >= xMid:
            right_points.append(points[i])
        if points[i][1] <= yMid:
            top_points.append(points[i])
        if points[i][1] >= yMid:
            bottom_points.append(points[i])

    left_points = np.array(left_points, dtype=np.float32)
    right_points = np.array(right_points, dtype=np.float32)
    top_points = np.array(top_points, dtype=np.float32)
    bottom_points = np.array(bottom_points, dtype=np.float32)

    left = fitting(left_points, dist, "V", size, step) 
    top = fitting(top_points, dist, "H", size, step) 
    right = fitting(right_points, dist, "V", size, step) 
    bottom = fitting(bottom_points, dist, "H", size, step)

    lt = intersect(left[0], left[1], left[2], top[0], top[1], top[2])
    rt = intersect(right[0], right[1], right[2], top[0], top[1], top[2])
    rb = intersect(right[0], right[1], right[2], bottom[0], bottom[1], bottom[2])
    lb = intersect(left[0], left[1], left[2], bottom[0], bottom[1], bottom[2])

    return np.array([lt, rt, rb, lb])

def bit2char(bits):
    result = ""
    value = 0
    for i in range(len(bits)):
        value = value << 1
        value = value + bits[(i//8)*8 + 7 - i%8]
        if i % 8 == 7:
            result += chr(value)
            value = 0
    return result

def main():
    args=argparse.ArgumentParser()
    args.add_argument("--video", "-v", dest="video", type=str, required=True)
    args.add_argument("--lightnet", "-ltn", dest="lightnetName", type=str, required=True)
    args.add_argument("--screennet", "-scn", dest="screennetName", type=str)
    args.add_argument("--interval","-t",dest="interval",type=int, default=10)
    args.add_argument("--otype","-o",dest="oType",type=int, default=1)
    args.add_argument("--numframe","-n",dest="numFrame",type=int, default=1000)
    args.add_argument("--debug", "-d", dest="debugOut", type=str)
    args=args.parse_args()

    video = cv2.VideoCapture(args.video)
    print("Frame count: " + str(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))
    if args.debugOut is not None:
        dbgVideo=cv2.VideoWriter(args.debugOut + ".avi",cv2.VideoWriter_fourcc('M','J','P','G'),5,(299,299))
        logFile = open(args.debugOut + ".txt", "w")
    
    #Load weight for model. Please use the deeplighttiny model
    print("================================= LIGHTNET =================================")
    lightnetmodel = lightnet.LightNetModel()
    lightnetmodel.load_weights(args.lightnetName)
    print(lightnetmodel.summary())
    print("================================= SCREENNET =================================")
    screennetmodel = screennet.ScreenNetModel()
    if args.screennetName is not None:
        screennetmodel.load_weights(args.screennetName)
    print(screennetmodel.summary())

    GRID_X = 10
    GRID_Y = 10
    #RS code preparation
    percentage = 50
    sym_bits = 5 #Number of bit per symbol
    block_len = 20 #Number of symbols in a block (we transmit data in a block)
    ratio = percentage / 100.0 #Ratio of data symbol in a block. The rest are error correct_frameion code
    msg_len = int(ratio*block_len) #Number of data symbols
    datalen = int(percentage*GRID_X*GRID_Y/100)
    codec = Codec(sym_bits=sym_bits, block_len=block_len, code_ratio=ratio)

    pre_seq_num = -1
    seq_num = 0
    
    ret0, frame0 = video.read()
    ret1, frame1 = video.read()
    b0, g0, r0 = cv2.split(frame0)
    b1, g1, r1 = cv2.split(frame1)

    #To test with different screen extraction error, please change the following corners (lefttop, righttop, rightbottom, leftbottom)
    scr_pos = np.array([(364, 123), (922, 118), (915, 435), (371, 435)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(scr_pos, np.array([(0,0),(298,0),(298,298),(0,298)], dtype=np.float32))
    fulltext = ""
    frame_index = 0
    while(True):
        ret2, frame2 = video.read()
        if not ret2:
            break
        if frame_index == args.numFrame:
            break
        
        if args.screennetName is not None:
            if frame_index % args.interval == 0: #Run  ScreenNet intermittently
                screennet_input = cv2.resize(frame2, (256, 256))
                screennet_input=(screennet_input.astype(np.float32)/128.0)-1.0
                screennet_input=np.reshape(screennet_input,(1,256,256,3))
                yPred=screennetmodel.predict(screennet_input)[0]
                yPred = 255*yPred/np.max(yPred)
                corners = findCorners(yPred.astype(np.uint8), 4, 40, 10)
                corners[:,0] = corners[:,0]*(frame2.shape[1]/256.0)
                corners[:,1] = corners[:,1]*(frame2.shape[0]/256.0)
                scr_pos = np.array([(corners[0,0], corners[0,1]), (corners[1,0], corners[1,1]), (corners[2,0], corners[2,1]), (corners[3,0], corners[3,1])], dtype=np.float32)
                transform = cv2.getPerspectiveTransform(scr_pos, np.array([(0,0),(298,0),(298,298),(0,298)], dtype=np.float32))
                if args.debugOut is not None:
                    dbgVideo.write(cv2.warpPerspective(frame2, transform, (299, 299)))

        b2, g2, r2 = cv2.split(frame2)
        tem = cv2.merge((b0, b1, b2))
        tem = cv2.warpPerspective(tem, transform, (299, 299))
        frame_in = tem.reshape((1, tem.shape[0], tem.shape[1], tem.shape[2])).astype(np.float32)
        frame_in /= 255
        frame_in -= 0.5
        frame_in *= 2.
        output = ((lightnetmodel.predict(frame_in)[0] >= 0.5)).astype(np.uint8)

        try:
            rs_preds = codec.decode(output)
            #Get the sequence number
            seq_num = 0
            for bit in range(6, 11):
                seq_num = seq_num << 1
                seq_num = seq_num + rs_preds[-bit]
            #get the checksum field
            csum_field = 0
            for bit in range(1, 6):
                csum_field = csum_field << 1
                csum_field = csum_field + rs_preds[-bit]
            #calculate the checksum from data
            csum_calc = 0
            for sym in range(int(datalen/sym_bits)-1):
                value = 0
                for bit in range(sym*sym_bits, sym*sym_bits+sym_bits):
                    value = value << 1
                    value += rs_preds[bit]
                csum_calc = ((csum_calc & 0x1F) >> 1) + ((csum_calc & 0x1) << 4)
                csum_calc = (csum_calc + value) & 0x1F
            #if checksum matched
            if csum_calc == csum_field and (np.sum(rs_preds.astype(np.uint8)) != 0):
                if pre_seq_num != seq_num:
                    if args.oType == 1:
                        txt = bit2char(rs_preds[0:40])
                        fulltext += txt
                        print(txt)
                    else:
                        txt = "SEQ: " + str(seq_num) + " | CSUM: " + str(csum_calc) + " | DATA: "
                        print(txt + ",".join([str(outbit) for outbit in rs_preds[0:40]]))
                        logFile.write(txt + ",".join([str(outbit) for outbit in rs_preds[0:40]]) + "\n")
                #else: Repeated
            # else:
                # print("======================== CSUM ERROR --> reject")
        except:
            pass
        cv2.imshow("Frame", frame_in[0])
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        b0 = b1
        b1 = b2
        pre_seq_num = seq_num
        frame_index += 1

    print(fulltext)
    video.release()
    if args.debugOut is not None:
        dbgVideo.release()
        logFile.close()

if __name__ == '__main__':
    main()
