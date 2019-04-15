#python screen-detection-1.py ./video/real00.avi

DEBUG=True
import sys
import cv2
import numpy as np
import sklearn

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial as P



def resizedFrame(frame,ratio):
    return cv2.resize(frame,(int(frame.shape[1]*ratio),int(frame.shape[0]*ratio)))


def mainFunc():
    inputFile=sys.argv[1]#"./video/real00.avi"
    outputFile="./tempOut.mp4"#sys.argv[2]

    sourcePoints="[[0,0], [1919,0], [1919,1070], [0,1079]]"#sys.argv[3]##
    destPoints="[[0,0], [299,0], [299,299], [0,299]]"#sys.argv[4]

    sourcePoints=np.array(eval(sourcePoints)).astype(np.float32)
    destPoints=np.array(eval(destPoints)).astype(np.float32)

    print("Source points shape = {}, dest = {}".format(sourcePoints.shape,destPoints.shape))

    vidIn=cv2.VideoCapture(inputFile)
    vidOut=cv2.VideoWriter(outputFile,cv2.VideoWriter_fourcc('M','J','P','G'),60,(299,299))



    while True:
        ret,frame=vidIn.read()
        assert ret , "No frame returned"
        frame=resizedFrame(frame,0.3)
        print("Frame shape{}".format(frame.shape))

        oriFrame=np.array(frame)

        topEdge=np.zeros((frame.shape[1]),dtype=np.int32)#.fill(-1)
        bottomEdge=np.zeros((frame.shape[1]),dtype=np.int32)#.fill(-1)
        leftEdge=np.zeros((frame.shape[0]),dtype=np.int32)#.fill(-1)
        rightEdge=np.zeros((frame.shape[0]),dtype=np.int32)#.fill(-1)

        THRESHOLD_LOW=1
        THRESHOLD_HIGH=20
        for x in range(frame.shape[1]):
            wentThroughBlack=False
            for y in range(frame.shape[0]):
                if not wentThroughBlack:
                    if np.mean(frame[y,x,:])<THRESHOLD_LOW:
                        wentThroughBlack = True
                else:
                    if np.mean(frame[y,x,:])>THRESHOLD_HIGH:
                        topEdge[x]=y

        for x in range(frame.shape[1]):
            wentThroughBlack=False
            for y in range(frame.shape[0]-1,-1,-1):
                if not wentThroughBlack:
                    if np.mean(frame[y,x,:])<THRESHOLD_LOW:
                        wentThroughBlack = True
                else:
                    if np.mean(frame[y,x,:])>THRESHOLD_HIGH:
                        bottomEdge[x]=y


        for y in range(frame.shape[0]):
            wentThroughBlack=False
            for x in range(frame.shape[1]):
                if not wentThroughBlack:
                    if np.mean(frame[y,x,:])<THRESHOLD_LOW:
                        wentThroughBlack = True
                else:
                    if np.mean(frame[y,x,:])>THRESHOLD_HIGH:
                        leftEdge[y]=x

        for y in range(frame.shape[0]):
            wentThroughBlack=False
            for x in range(frame.shape[1]-1,-1,-1):
                if not wentThroughBlack:
                    if np.mean(frame[y,x,:])<THRESHOLD_LOW:
                        wentThroughBlack = True
                else:
                    if np.mean(frame[y,x,:])>THRESHOLD_HIGH:
                        rightEdge[y]=x


        PARAM_1=0.2
        PARAM_2=0.8

        x=np.arange(int(topEdge.shape[0]*PARAM_1),int(topEdge.shape[0]*PARAM_2))
        y=topEdge[np.arange(int(topEdge.shape[0]*PARAM_1),int(topEdge.shape[0]*PARAM_2))]
        topLine=P.polyfit(x,y,1)

        x=np.arange(int(bottomEdge.shape[0]*PARAM_1),int(bottomEdge.shape[0]*PARAM_2))
        y=bottomEdge[np.arange(int(bottomEdge.shape[0]*PARAM_1),int(bottomEdge.shape[0]*PARAM_2))]
        bottomLine=P.polyfit(x,y,1)

        y=np.arange(int(rightEdge.shape[0]*PARAM_1),int(rightEdge.shape[0]*PARAM_2))
        x=rightEdge[np.arange(int(rightEdge.shape[0]*PARAM_1),int(rightEdge.shape[0]*PARAM_2))]
        rightLine=P.polyfit(x,y,1)


        y=np.arange(int(leftEdge.shape[0]*PARAM_1),int(leftEdge.shape[0]*PARAM_2))
        x=leftEdge[np.arange(int(leftEdge.shape[0]*PARAM_1),int(leftEdge.shape[0]*PARAM_2))]
        leftLine=P.polyfit(x,y,1)

        if DEBUG:
            print("Top line ",topLine)
            print("Bottom line ",bottomLine)
            print("Right line ",rightLine)
            print("Left line ",leftLine)


        '''
            p1          p2
            
            
            p4          p3
        '''

        p1=P.polyroots(topLine-leftLine)
        p2=P.polyroots(topLine-rightLine)
        p3=P.polyroots(rightLine-bottomLine)
        p4=P.polyroots(bottomLine-leftLine)

        if DEBUG: print("Points ",p1,p2,p3,p4)
        if DEBUG: print("P1 ",P.polyval(p1,topLine),P.polyval(p1,leftLine))
        if DEBUG: print("P2 ",P.polyval(p2,topLine),P.polyval(p2,rightLine))
        if DEBUG: print("P3 ",P.polyval(p3,rightLine),P.polyval(p3,bottomLine))
        if DEBUG: print("P4 ",P.polyval(p4,leftLine),P.polyval(p4,bottomLine))

        p1xy=np.array([p1,P.polyval(p1,topLine)])
        p2xy=np.array([p2,P.polyval(p2,topLine)])
        p3xy=np.array([p3,P.polyval(p3,bottomLine)])
        p4xy=np.array([p4,P.polyval(p4,bottomLine)])


        cv2.circle(frame,(p1xy[1],p1xy[0]),3,[255,0,0])
        cv2.circle(frame,(p2xy[1],p2xy[0]),3,[255,0,0])
        cv2.circle(frame,(p3xy[1],p3xy[0]),3,[255,0,0])
        cv2.circle(frame,(p4xy[1],p4xy[0]),3,[255,0,0])



        if DEBUG:
            for i in range(max(frame.shape[0],frame.shape[1])):
                try:
                    cv2.circle(frame,(leftEdge[i],i),3,[255,0,0])
                    cv2.circle(frame,(rightEdge[i],i),3,[0,255,0])
                    cv2.circle(frame,(i,topEdge[i]),3,[0,0,255])
                    cv2.circle(frame,(i,bottomEdge[i]),3,[255,255,255])
                except:
                    None



        if DEBUG:
            '''print("Left edge",leftEdge)
            print("Right edge",rightEdge)
            print("Top edge",topEdge)
            print("Bottom edge",bottomEdge)'''

        cv2.imshow("Result",frame)
        cv2.imshow("Original frame",oriFrame)

        cv2.waitKey(10)

    '''frameFlat=np.reshape(frame,(frame.shape[0]*frame.shape[1],frame.shape[2]))
    print(frame.shape,frameFlat.shape)

    k=sklearn.cluster.KMeans(n_clusters=10)
    k.fit(frameFlat)'''

    '''cv2.imshow("Input image",resizedFrame(frame,1.0))
    cv2.waitKey(10000)

    h=cv2.getPerspectiveTransform(sourcePoints,destPoints)
    destFrame = cv2.warpPerspective(frame, h, (299,299))'''



    cv2.destroyAllWindows()
    vidIn.release()
    vidOut.release()


if __name__ == '__main__':
    mainFunc()



