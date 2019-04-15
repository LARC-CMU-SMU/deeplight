#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
gihanchanaka@gmail.com
07-03-2019
    1)This is to learn from BW videos
'''


# In[2]:


DEBUG=False
PC=False


# In[3]:


FILE_NAME='./video/ran'
FILE_NAME_VIDEO=FILE_NAME+'.avi'
FILE_NAME_CSV=FILE_NAME+'.csv'


# In[4]:


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


# In[5]:


keras.callbacks.TensorBoard(log_dir='./logs/bw/', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,)


# In[6]:


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





# In[7]:


NO_FRAMES=10000
if PC: NO_FRAMES=100
FRAMES_TO_SKIP=0
'''NO_OF_FRAMES=int(sys.argv[1])
'''
FRAME_HEIGHT=108
FRAME_WIDTH=192
COLOR_CHANNELS=3

CELLS_PER_FRAME=4

EPOCHS=20
BATCH_SIZE=64
CUDA1=0
CUDA2=1

os.environ["CUDA_VISIBLE_DEVICES"]="{},{}".format(CUDA1,CUDA2)


# In[8]:


def makeX():
    cap = cv2.VideoCapture(FILE_NAME_VIDEO)
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    ret,frame=cap.read()
    '''if (len(sys.argv)>1):
        NO_OF_FRAMES=int(sys.argv[1])'''

    X=np.zeros((NO_FRAMES,FRAME_HEIGHT,FRAME_WIDTH,COLOR_CHANNELS),dtype=np.float32)
        
    for f in range(NO_FRAMES):
        X[f,:,:,:]=frame
        ret,frame=cap.read()

        '''if f%1 == 0:
            print(len(X[f,:,4,1]))'''
    
    return X
#maxeX(cap)


# In[9]:


def makeY():
    df=pd.read_csv(FILE_NAME_CSV, sep=',',header=None)
    df=np.array(df)
    return df[:NO_FRAMES,:]
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
    


# In[10]:


X=makeX()
Y=makeY()
X=X[FRAMES_TO_SKIP:]
Y=Y[FRAMES_TO_SKIP:]

if DEBUG: print("X= ",X)

if DEBUG: print("NAN in X",np.argwhere(np.isnan(X)))
if DEBUG: print("NAN in Y",np.argwhere(np.isnan(Y)))


if DEBUG: print("X shape: ",np.shape(X))
if DEBUG: print("Y shape: ",np.shape(Y))

INPUT_DIM=X[0].shape
OUTPUT_DIM=CELLS_PER_FRAME


# In[8]:


'''Preparing the input and output vectors'''
X=np.mean(X,axis=3)
X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))

X=(X-128.0)/256.0

print("X shape {}, Y shape {}.".format(X.shape,Y.shape))

xTrain, xTest, yTrain, yTest= train_test_split(X, Y, test_size=0.01)


# In[ ]:


'''countY=np.zeros(2**CELLS_PER_FRAME)
for f in range(yTrain.shape[0]):
    num=0
    for m in range(CELLS_PER_FRAME):
        num=num*2+yTrain[f][m]
    countY[num]=countY[num]+1

for c in range(2**CELLS_PER_FRAME):
    print(c,countY[c])'''


# In[10]:


if PC:
    cv2.imshow('Example frame',X[10,:,:,:])
    cv2.waitKey(100)
    print(Y[10])


# In[ ]:


def maxDiff(y_true, y_pred):
    return K.max(K.abs(y_true-y_pred))


# In[11]:


model=cnn(INPUT_DIM,OUTPUT_DIM)
model.build()
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy','accuracy','mean_squared_error','binary_crossentropy',maxDiff])
model.fit(xTrain,yTrain,validation_split=0.1,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE,shuffle=False)
print(model.summary())


print("Test set evaluation",model.evaluate(xTest,yTest))

print(np.round(model.predict(xTest)))


# In[ ]:




