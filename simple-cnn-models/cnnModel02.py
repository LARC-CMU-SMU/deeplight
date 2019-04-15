#!/usr/bin/env python
# coding: utf-8

# In[1]:


DEBUG=Flase


# In[2]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

import numpy as np
import cv2
import sys
import os


# In[3]:


cap = cv2.VideoCapture('./video/none_256_5_mix.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


# In[4]:


NO_OF_FRAMES=1010
FRAMES_TO_SKIP=10
'''NO_OF_FRAMES=int(sys.argv[1])
'''
FRAME_HEIGHT=1080
FRAME_WIDTH=1920
COLOR_CHANNELS=3

CELLS_PER_FRAME=9

EPOCHS=10
BATCH_SIZE=64
CUDA1=0
CUDA2=1

os.environ["CUDA_VISIBLE_DEVICES"]="{},{}".format(CUDA1,CUDA2)


'''EPOCHS=int(sys.argv[2])
BATCH_SIZE=int(sys.argv[3])
CUDA1=int(sys.argv[4])
CUDA2=int(sys.argv[5])'''


# In[5]:


'''while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break'''


# In[6]:


def makeX(cap):
    ret,frame=cap.read()
    '''if (len(sys.argv)>1):
        NO_OF_FRAMES=int(sys.argv[1])'''

    X=np.zeros((NO_OF_FRAMES,FRAME_HEIGHT,FRAME_WIDTH,COLOR_CHANNELS))
        
    for f in range(NO_OF_FRAMES):
        X[f,:,:,:]=frame
        ret,frame=cap.read()

        '''if f%1 == 0:
            print(len(X[f,:,4,1]))'''
    
    return X
#maxeX(cap)


# In[7]:


def makeY():
    import pandas as pd
    df=pd.read_csv('./video/none_256_5_mix.csv', sep=',',header=None)
    df=np.array(df)
    df=df[:NO_OF_FRAMES,1]
    
    Y=np.zeros((NO_OF_FRAMES,CELLS_PER_FRAME),dtype=np.int32)
    
    for f in range(NO_OF_FRAMES):
        if DEBUG: print("DEBUG: df=",df[f])
        for c in range(CELLS_PER_FRAME-1,-1,-1):
            Y[f,c]=int(df[f]/(2**c))
            if Y[f,c]>0:
                df[f]=df[f]-2**c
        
        if DEBUG: print("DEBUG: Y=",Y[f,:])
    
    return Y
    
#makeY()


# In[8]:


X=makeX(cap)
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


# In[10]:


'''Preparing the input and output vectors'''
X=np.mean(X,axis=3)
X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))


print("X shape {}, Y shape {}.".format(X.shape,Y.shape))

xTrain, xTest, yTrain, yTest= train_test_split(X, Y, test_size=0.2)


# In[23]:


def cnn(inptuDim,outputDim):
    model=Sequential()
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Flatten())
    #model.add(Dense(500,activation='relu'))
    #model.add(Dense(81,activation='sigmoid'))
    model.add(Dense((CELLS_PER_FRAME),activation='sigmoid'))

    return model


# In[24]:


model=cnn(INPUT_DIM,OUTPUT_DIM)
model.build()
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(xTrain,yTrain,validation_split=0.3,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE,shuffle=False)
print(model.summary())



# In[15]:


print(model.evaluate(xTest,yTest,batch_size=BATCH_SIZE))


# In[14]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:




