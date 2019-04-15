#!/usr/bin/env python
# coding: utf-8

# In[1]:


DEBUG=True
DEBUG_SHOW_VIDEO=False
PC=False
GPU=1
NO_MAX_FRAMES=20000


TARGET_COST=0.01


# In[2]:


import tensorflow as tf
tf.set_random_seed(625742)
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import keras.backend as K
import time
import numpy as np
import cv2
import sys
import os
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.models import Model
from keras.utils import multi_gpu_model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.optimizers import rmsprop
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image


# In[3]:


def makeY(fileNameCsv,noFrames,skipFrames):
    df=pd.read_csv(fileNameCsv, sep=',',header=None)
    df=np.array(df).astype(np.float32)
    
    return df[skipFrames:noFrames,:]


# In[4]:



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


# In[5]:


FRAME_HEIGHT=360
FRAME_WIDTH=480
COLOR_CHANNELS=3
CELLS_PER_FRAME=400




INPUT_DIM=(FRAME_HEIGHT,FRAME_WIDTH)
OUTPUT_DIM=CELLS_PER_FRAME


EPOCHS=200
BATCH_SIZE=64
CUDA1=4
CUDA2=5


if GPU==0:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
elif GPU==1:
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(CUDA1)
elif GPU==2:
    os.environ["CUDA_VISIBLE_DEVICES"]="{},{}".format(CUDA1,CUDA2)

sess = tf.Session()


# In[7]:


def trainAndTestForVideo(model,fileName,noFrames,framesToSkip=0,videoFileFormat='.avi',testSplit=0.1):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Starting new video file\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}".format(fileName))
    FILE_NAME=fileName
    FILE_NAME_VIDEO=FILE_NAME+videoFileFormat
    
    if videoFileFormat=='.npz':
        data=np.load(FILE_NAME_VIDEO)
        dataX=data['X'][framesToSkip:noFrames]
        dataY=data['Y'][framesToSkip:noFrames]
    else:
        
        FILE_NAME_CSV=FILE_NAME+'.csv'

        NO_FRAMES=noFrames
        if PC: NO_FRAMES=100
        FRAMES_TO_SKIP=framesToSkip

        dataX=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
        dataY=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)
    
    X=np.ndarray((dataX.shape[0],299,299,3),np.float32)
    
    for f in range(dataX.shape[0]):
        X[f]=cv2.resize(dataX[f],(299,299),interpolation = cv2.INTER_LINEAR)/np.float32(256.0)
    
    dataX=None
    
    if DEBUG_SHOW_VIDEO:
        for i in range(X.shape[0]):
            cv2.imshow("Image",X[i])
            cv2.waitKey(10)
        cv2.destroyAllWindows()
    
    
    hist={"trainAcc":[],"trainCost":[],"testAcc":[],"testCost":[]}
    if DEBUG: print("X type: {}, Y type: {}.".format(X.dtype,dataY.dtype))
    xTrain, xTest, yTrain, yTest= train_test_split(X, dataY, test_size=testSplit)
    X=None
    dataY=None
    print("SIZES: xTrain {}, yTrain {}, xTest {}, yTest {}".format(xTrain.shape,yTrain.shape,xTest.shape,yTest.shape))
    
    iters=int(input("How many iters more? : "))
    while iters>0:
        model.fit(xTrain,yTrain,epochs=1, verbose=1, shuffle=True)#,batch_size=BATCH_SIZE)
        
        yTrainPredFloat=model.predict(xTrain)
        yTestPredFloat=model.predict(xTest)
        yTrainPred=np.round(yTrainPredFloat)
        yTestPred=np.round(yTestPredFloat)
        trainAcc=np.mean(np.mean((yTrainPred==yTrain).astype(np.float32)))
        testAcc=np.mean(np.mean((yTestPred==yTest).astype(np.float32)))
        trainCost=np.mean(np.mean(np.power(yTrainPredFloat-yTrain,2)))
        testCost=np.mean(np.mean(np.power(yTestPredFloat-yTest,2)))
        
        
        hist["trainAcc"].append(trainAcc)
        hist["trainCost"].append(trainCost)
        hist["testAcc"].append(testAcc)
        hist["testCost"].append(testCost)

        
        iters-=1
        if iters==0:
            iters=int(input("How many iters more? : "))
        
    print(hist)


# In[8]:


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)

    return x


# In[9]:


def inceptionBlockA(x,noKernels):
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, noKernels, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],axis=3)#,name='mixed0{}'.format(l))

    return x


# In[10]:


def inceptionBlockB(x,noKernels):
    branch1x1 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(x, noKernels, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, noKernels, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, noKernels, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, noKernels, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, noKernels, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, noKernels, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate( [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)#,name='mixed4{}'.format(l))
    return x


# In[11]:


def inceptionBlockC(x):
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = layers.concatenate( [branch3x3_1, branch3x3_2], axis=3)#,name='mixed9_' + str(i))

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate( [branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3)#,
    return x


# In[12]:


def inceptionConeA(img_input):

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    return x


# In[13]:


def inceptionConeB(x):
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn( branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate( [branch3x3, branch3x3dbl, branch_pool], axis=3)#, name='mixed3')
    return x


# In[14]:


def inceptionConeC(x):
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate( [branch3x3, branch7x7x3, branch_pool], axis=3)#, name='mixed8')
    return x


# In[15]:


def inceptionNet():

    imgInput = Input(shape=(299,299,3))
    
    print(imgInput.shape)
    x=inceptionConeA(imgInput)
    print(x.shape)
    #x=inceptionBlockA(x,32)
    #print("After 1A",x.shape)
    x=inceptionBlockA(x,64)
    print("After 2A",x.shape)
    #x=inceptionBlockA(x,64)
    #print("After 3A",x.shape)
    x=inceptionConeB(x)
    print(x.shape)
    #x=inceptionBlockB(x,128)
    #print("After 1B",x.shape)
    #x=inceptionBlockB(x,160)
    #print("After 2B",x.shape)
    #x=inceptionBlockB(x,160)
    #print("After 3B",x.shape)
    x=inceptionBlockB(x,192)
    print("After 4B",x.shape)
    x=inceptionConeC(x)
    print(x.shape)
    #x=inceptionBlockC(x)
    #print("After 1C",x.shape)
    x=inceptionBlockC(x)
    print("After 2C",x.shape)
    
    
    x = AveragePooling2D(pool_size=(8, 8), strides=(8,8),name='avg_pool')(x)
    print(x.shape)
    x = Flatten()(x)
    print(x.shape)
    x = Dropout(0.25)(x)
    print(x.shape)
    '''x = Dense(800)(x)
    print(x.shape)'''
    x = Dense(CELLS_PER_FRAME, activation='sigmoid', name='predictions')(x)
    print(x.shape)
    
    model = Model(imgInput, x, name='inception_v3')
    opt = rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])
    
    print(model.summary())
    return model


# In[16]:


trainAndTestForVideo(inceptionNet(),'./video/butterfly-20x20',7500,videoFileFormat='.npz',testSplit=0.2)


# In[ ]:





# In[ ]:




