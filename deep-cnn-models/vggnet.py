#!/usr/bin/env python
# coding: utf-8

# In[ ]:


DEBUG=True
PC=False
GPU=1
NO_MAX_FRAMES=20000


TARGET_COST=0.01


# In[ ]:


import tensorflow as tf
tf.set_random_seed(625742)
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
# import keras.backend as K
import time

import numpy as np
import cv2
import sys
import os
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np


# In[ ]:


def makeY(fileNameCsv,noFrames,skipFrames):
    df=pd.read_csv(fileNameCsv, sep=',',header=None)
    df=np.array(df).astype(np.float32)
    
    return df[skipFrames:noFrames,:]


# In[ ]:



def makeX(fileNameVideo,noFrames,skipFrames):
    cap = cv2.VideoCapture(fileNameVideo)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    ret,frame=cap.read()


    X=np.zeros((noFrames,FRAME_HEIGHT,FRAME_WIDTH,COLOR_CHANNELS),dtype=np.float32)
        
    for f in range(noFrames):
        X[f,:,:,:]=frame
        ret,frame=cap.read()
    return X[skipFrames:,:,:,:]


# In[ ]:


FRAME_HEIGHT=224
FRAME_WIDTH=224
COLOR_CHANNELS=3
CELLS_PER_FRAME=9

INPUT_DIM=(FRAME_HEIGHT,FRAME_WIDTH)
OUTPUT_DIM=CELLS_PER_FRAME


EPOCHS=200
BATCH_SIZE=64
CUDA1=4
CUDA2=7


if GPU==0:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
elif GPU==1:
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(CUDA1)
elif GPU==2:
    os.environ["CUDA_VISIBLE_DEVICES"]="{},{}".format(CUDA1,CUDA2)

sess = tf.Session()


# In[ ]:


def trainAndTestForVideo(model,fileName,noFrames,framesToSkip=0,videoFileFormat='.avi',testSplit=0.1):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Starting new video file\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}".format(fileName))
    
    FILE_NAME=fileName
    FILE_NAME_VIDEO=FILE_NAME+videoFileFormat
    FILE_NAME_CSV=FILE_NAME+'.csv'

    NO_FRAMES=noFrames
    if PC: NO_FRAMES=100
    FRAMES_TO_SKIP=framesToSkip

    dataX=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
    dataY=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)
    
    dataX=(dataX-127.5)/128.0

    
    if DEBUG: print("X type: {}, Y type: {}.".format(dataX.dtype,dataY.dtype))
    xTrain, xTest, yTrain, yTest= train_test_split(dataX, dataY, test_size=testSplit)
    print("SIZES: xTrain {}, yTrain {}, xTest {}, yTest {}".format(xTrain.shape,yTrain.shape,xTest.shape,yTest.shape))
    
    iters=int(input("How many iters more? : "))
    while iters>0:
        model.fit(xTrain,yTrain,epochs=1, verbose=1)#,batch_size=BATCH_SIZE)
        iters-=1
        if iters==0:
            iters=int(input("How many iters more? : "))
        
    print(model.evaluate(xTrain,yTrain))
    print(model.evaluate(xTest,yTest))
    
    
    


# In[ ]:



def vggNet():

    #Instantiate an empty model
    model = Sequential([
    
    Conv2D(3, (1, 1), input_shape=(FRAME_HEIGHT,FRAME_WIDTH,COLOR_CHANNELS), padding='same', activation='relu'),
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     Conv2D(128, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
#     Dense(256, activation='relu'),
    Dense(CELLS_PER_FRAME, activation='sigmoid')
    ])

    '''Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),'''
    
    model.summary()

    model.compile(loss=keras.losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])
    
    return model


# In[ ]:


trainAndTestForVideo(vggNet(),'./video/real00',10000,videoFileFormat='.avi',testSplit=0.2)


# In[ ]:




