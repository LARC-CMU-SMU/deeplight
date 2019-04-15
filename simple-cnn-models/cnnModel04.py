#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
gihanchanaka@gmail.com
08-03-2019
    1)This is to learn from BW and predict random
    2)          learn from random and predict movie
'''


# In[ ]:


DEBUG=False
PC=False


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Dropout
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import keras.backend as K

import numpy as np
import cv2
import sys
import os
import pandas as pd


# In[ ]:


keras.callbacks.TensorBoard(log_dir='./logs/cnnModel04/', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,)


# In[ ]:


FRAME_HEIGHT=100
FRAME_WIDTH=200
COLOR_CHANNELS=3
CELLS_PER_FRAME=25

INPUT_DIM=(FRAME_HEIGHT,FRAME_WIDTH)
OUTPUT_DIM=CELLS_PER_FRAME


EPOCHS=20
BATCH_SIZE=64
CUDA1=0
CUDA2=1

os.environ["CUDA_VISIBLE_DEVICES"]="{},{}".format(CUDA1,CUDA2)


# In[ ]:


def cnn(inptuDim,outputDim):
    model=Sequential()
    '''model.add(Conv2D(10, kernel_size=(10,10), activation='relu',strides=(10,10)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))'''
    model.add(Flatten())
    #model.add(Dense(500,activation='sigmoid'))
    #model.add(Dropout(0.5))
    #model.add(Dense(81,activation='relu'))
    model.add(Dense((CELLS_PER_FRAME),activation='sigmoid'))

    return model


# In[ ]:


def makeX(fileNameVideo,noFrames,skipFrames):
    cap = cv2.VideoCapture(fileNameVideo)
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    ret,frame=cap.read()
    '''if (len(sys.argv)>1):
        NO_OF_FRAMES=int(sys.argv[1])'''

    X=np.zeros((noFrames,FRAME_HEIGHT,FRAME_WIDTH,COLOR_CHANNELS),dtype=np.float32)
        
    for f in range(NO_FRAMES):
        X[f,:,:,:]=frame
        ret,frame=cap.read()

        '''if f%1 == 0:
            print(len(X[f,:,4,1]))'''
    
    return X[skipFrames:,:,:,:]
#maxeX(cap)


# In[ ]:


def makeY(fileNameCsv,noFrames,skipFrames):
    df=pd.read_csv(fileNameCsv, sep=',',header=None)
    df=np.array(df)
    return df[skipFrames:noFrames,:]
    '''df=df[:NO_OF_FRAMES,1]
    
    Y=np.zeros((NO_OF_FRAMES,CELLS_PER_FRAME),dtype=np.int32)
    100
    for f in range(NO_OF_FRAMES):
        if DEBUG: print("DEBUG: df=",df[f])
        for c in range(CELLS_PER_FRAME-1,-1,-1):
            Y[f,c]=int(df[f]/(2**c))
            if Y[f,c]>0:100
                df[f]=df[f]-2**c
        
        if DEBUG: print("DEBUG: Y=",Y[f,:])
    
    return Y'''
    


# In[ ]:


def maxDiff(y_true, y_pred):
    return K.max(K.abs(y_true-y_pred))


# In[ ]:


model=cnn(INPUT_DIM,OUTPUT_DIM)
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy','accuracy','mean_squared_error','binary_crossentropy',maxDiff])
#model.predict(xTest,yTest)
#print(model.summary())




# In[ ]:


'''
BLACK AND WHITE VIDEO

'''
FILE_NAME='./video/bw'
FILE_NAME_VIDEO=FILE_NAME+'.avi'
FILE_NAME_CSV=FILE_NAME+'.csv'

NO_FRAMES=10000
if PC: NO_FRAMES=100
FRAMES_TO_SKIP=0


# In[ ]:


X=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
Y=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)

if DEBUG: print("X= ",X)
if DEBUG: print("NAN in X",np.argwhere(np.isnan(X)))
if DEBUG: print("NAN in Y",np.argwhere(np.isnan(Y)))
if DEBUG: print("X shape: ",np.shape(X))
if DEBUG: print("Y shape: ",np.shape(Y))


# In[ ]:


'''Preparing the input and output vectors'''
X=np.mean(X,axis=3)
X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
X=(X-128.0)/256.0

print("X shape {}, Y shape {}.".format(X.shape,Y.shape))
xTrain, xTest, yTrain, yTest= train_test_split(X, Y, test_size=0.01)


# In[ ]:


model.fit(xTrain,yTrain,validation_split=0.1,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE,shuffle=True)
print("Test set evaluation for BW video",'\n',model.metrics_names,'\n', model.evaluate(xTest,yTest))


# In[ ]:


'''
RANDOM COLOUR VIDEO

'''
FILE_NAME='./video/ran'
FILE_NAME_VIDEO=FILE_NAME+'.avi'
FILE_NAME_CSV=FILE_NAME+'.csv'

NO_FRAMES=10000
if PC: NO_FRAMES=100
FRAMES_TO_SKIP=0


# In[ ]:


X=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
Y=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)

if DEBUG: print("X= ",X)
if DEBUG: print("NAN in X",np.argwhere(np.isnan(X)))
if DEBUG: print("NAN in Y",np.argwhere(np.isnan(Y)))
if DEBUG: print("X shape: ",np.shape(X))
if DEBUG: print("Y shape: ",np.shape(Y))

INPUT_DIM=X[0].shape
OUTPUT_DIM=CELLS_PER_FRAME


# In[ ]:


'''Preparing the input and output vectors'''
X=np.mean(X,axis=3)
X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
X=(X-128.0)/256.0

print("X shape {}, Y shape {}.".format(X.shape,Y.shape))
xTrain, xTest, yTrain, yTest= train_test_split(X, Y, test_size=0.10)


# In[ ]:


model.fit(xTrain,yTrain,validation_split=0.1,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE,shuffle=True)
print("Test set evaluation for random video",'\n',model.metrics_names,'\n', model.evaluate(xTest,yTest))


# In[ ]:


'''
MOVIE VIDEO

'''
FILE_NAME='./video/multipleVideos'
FILE_NAME_VIDEO=FILE_NAME+'.mp4'
FILE_NAME_CSV=FILE_NAME+'.csv'

NO_FRAMES=10000
if PC: NO_FRAMES=100
FRAMES_TO_SKIP=0


# In[ ]:


X=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
Y=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)

if DEBUG: print("X= ",X)
if DEBUG: print("NAN in X",np.argwhere(np.isnan(X)))
if DEBUG: print("NAN in Y",np.argwhere(np.isnan(Y)))
if DEBUG: print("X shape: ",np.shape(X))
if DEBUG: print("Y shape: ",np.shape(Y))

INPUT_DIM=X[0].shape
OUTPUT_DIM=CELLS_PER_FRAME


# In[ ]:


'''Preparing the input and output vectors'''
X=np.mean(X,axis=3)
X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
X=(X-128.0)/256.0

print("X shape {}, Y shape {}.".format(X.shape,Y.shape))
xTrain, xTest, yTrain, yTest= train_test_split(X, Y, test_size=0.5)


# In[ ]:


print("Test set evaluation for Movie video (Without training for movie video): ",model.metrics_names,'  ', model.evaluate(xTest,yTest))
hist= model.fit(xTrain,yTrain,validation_split=0.1,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE,shuffle=True)
print("Training history for multiple videos\n",hist.history)
print("Test set evaluation for Movie video (After training for movie video):",model.metrics_names,'  ', model.evaluate(xTest,yTest))


# In[ ]:


print(model.summary())


# In[ ]:




