#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
gihanchanaka@gmail.com
11-03-2019
    1)This is to learn from BW and predict random
    2)          learn from random and predict movie
'''


# In[2]:


DEBUG=True
PC=False

TARGET_COST=0.001


# In[3]:


# from keras.models import Sequential
# from keras.layers import Conv2D,Flatten,Dense,Dropout
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
# import keras.backend as K

import numpy as np
import cv2
import sys
import os
import pandas as pd


# In[4]:


currentTotalIterations=0
#keras.callbacks.TensorBoard(log_dir='./logs/cnnModel04/', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,)


# In[5]:


def inputLayer(inputTensor,outputShape=None):
    if outputShape==None:
        outputTensor= inputTensor
    else:
        #x2D=tf.reshape(xFlat,[-1,imgSize,imgSize,noColorChannels])
        inputShape=inputTensor.get_shape()
        outputTensor=tf.reshape(inputTensor,[-1,outputShape[0],outputShape[1],1])
    
    if DEBUG: print("DEBUG: Conv layer added with output tensor shape {}".format(outputTensor.shape))
    return outputTensor

    


# In[6]:


def convLayer2D(inputTensor,kernelShape,noKernels,poolingSize=None,activation=None):
    filterLength=kernelShape[0]
    filterHeight=kernelShape[1]
    noFilters=noKernels
    inputShape=inputTensor.get_shape()
    if DEBUG: print("DEBUG: {}".format(inputShape))
    noChannels=int(inputShape[-1])

    shape=[filterLength,filterHeight,noChannels,noFilters]
    if DEBUG: print("DEBUG: Shape of weights {}".format(shape))

    weights=tf.Variable(tf.truncated_normal(shape,stddev=0.5))
    biases=tf.Variable(tf.constant(0.05,shape=[noFilters]))
    outputTensor=tf.nn.conv2d(input=inputTensor,filter=weights,strides=[1,1,1,1],padding='SAME')
    #strides=[img,x,y,colourChannel]
    outputTensor=outputTensor+biases
    
    if poolingSize!=None:
        outputTensor=maxPoolingLayer(outputTensor,poolingSize)
    if activation=='relu':
        outputTensor=relu(outputTensor)
    if DEBUG: print("DEBUG: Conv layer added with output tensor shape {}".format(outputTensor.shape))
    return outputTensor



# In[7]:


#def maxPoolingLayer(inputTensor,kernelSize):
    return tf.nn.max_pool(value=inputTensor,ksize=[1,kernelSize[0],kernelSize[1],1],strides=[1,kernelSize[0],kernelSize[1],1],padding='SAME')


# In[8]:


def relu(inputTensor):    
    return tf.nn.relu(inputTensor)


# In[9]:


def sigmoid(inputTensor):    
    return tf.nn.sigmoid(inputTensor)


# In[10]:


def flatten(inputTensor):
    shape=inputTensor.get_shape()  # layer_shape == [num_images, img_height, img_width, num_channels]
    noFeatures=int(shape[1]*shape[2]*shape[3])
    outputTensor=tf.reshape(inputTensor,[-1,noFeatures])
    return outputTensor


# In[11]:


def fullyConnectedLayer(inputTensor,outputSize):
    inputShape=inputTensor.get_shape()
    shape=[int(inputShape[1]),outputSize]
    weights=tf.Variable(tf.truncated_normal(shape,stddev=0.5))
    biases=tf.Variable(tf.constant(0.05,shape=[outputSize]))
    outputTensor=tf.matmul(inputTensor,weights)+biases
    if DEBUG: print("DEBUG: Added fully connected layer with output shape {}.".format(outputTensor.shape))
    return outputTensor


# In[12]:


def buildNetwork(outputTensor,targetTensor):
    if DEBUG: print("DEBUG: Building network target tensor {} out tensor {}".format(targetTensor.shape,outputTensor.shape))
    sqError=tf.squared_difference(outputTensor,targetTensor)  
    if DEBUG: print("DEBUG: sqError shape= {}".format(sqError.shape))
    cost=tf.reduce_mean(sqError,1)
    cost=tf.reduce_mean(cost,0)
    if DEBUG: print("DEBUG: cost shape= {}".format(cost.shape))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    yPred=tf.round(outputTensor)
    if DEBUG: print("DEBUG: yPred shape= {}".format(yPred.shape))
    correctPrediction=tf.equal(yPred,targetTensor)
    if DEBUG: print("DEBUG: correctPrediction shape= {}".format(correctPrediction.shape))
    accuracy=tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
    if DEBUG: print("DEBUG: accuracy shape= {}".format(accuracy.shape))
    
    
    print("yPred",yPred.shape,"Correct pred",correctPrediction.shape,"Accuracy",accuracy.shape,"Sq error",sqError.shape,"Cost",cost.shape)    
    
    if DEBUG: print("DEBUG: Completed building network")
    return outputTensor,optimizer,accuracy,cost


# In[23]:


def train(iterr,optimizer,noIterations,X,Y):
    DISPLAY_PROGRESS_EVERY_ITER=1
    DISPLAY_PROGRESS_EVERY_BATCH=25
    
    global currentTotalIterations
    
    
    for it in range(currentTotalIterations,currentTotalIterations+noIterations):
        sess.run(iterr.initializer,feed_dict={placeX:X,placeY:Y})
        
        batch=0
        while True:
#            sess.run(optimizer)
             try:
                 sess.run(optimizer)
#                 '''if batch%DISPLAY_PROGRESS_EVERY_BATCH==0:
#                     #if batch != 0: sys.stdout.write("\r")
#                     sys.stdout.write("Iter= {} ,Batch= {}".format(it+1,batch))
#                     sys.stdout.flush()
#                 batch+=1'''
                 batch+=1
             except:
                 print("{} batches trained",batch)
                 break 
        
        if (it>0)  and (it%DISPLAY_PROGRESS_EVERY_ITER==0):
            print("Iter {} complete".format(it+1))
            #printAccuracy(iterr,dataset,acc,"Train",cst)
    currentTotalIterations+=noIterations
    print("DEBUG: train() function completed")


# In[14]:


def makeY(fileNameCsv,noFrames,skipFrames):
    df=pd.read_csv(fileNameCsv, sep=',',header=None)
    df=np.array(df).astype(np.float32)
    
    return df[skipFrames:noFrames,:]


# In[15]:


def makeX(fileNameVideo,noFrames,skipFrames):
    cap = cv2.VideoCapture(fileNameVideo)
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    ret,frame=cap.read()


    X=np.zeros((noFrames,FRAME_HEIGHT,FRAME_WIDTH,COLOR_CHANNELS),dtype=np.float32)
        
    for f in range(NO_FRAMES):
        X[f,:,:,:]=frame
        ret,frame=cap.read()
    return X[skipFrames:,:,:,:]


# In[16]:


def maxDiff(y_true, y_pred):
    return K.max(K.abs(y_true-y_pred))


# In[17]:


def printAccuracy(iterr,X,Y,acc,cost,trainOrTest):
    DISPLAY_PROGRESS_EVERY_BATCH=25

    sess.run(iterr.initializer,feed_dict={placeX:X,placeY:Y})
    
    accCountProductSum=0.0
    costCountProductSum=0.0
    countSum=0.0
    batches=0
    while True:
        try:
            
            #Please note: These two lines are not 100% correct. This is just guesswork!
            a,c=sess.run([acc,cost])
            accCountProductSum+=a
            costCountProductSum+=c
#            if DEBUG: print("DEBUG: Batch accuracy= {}, batch cost= {}".format(a,c))
            countSum+=1
            batches+=1
            
#             if batches%DISPLAY_PROGRESS_EVERY_BATCH==0:
#                     #if batches != 0: sys.stdout.write("\r")
#                     sys.stdout.write("Evaluating batch= {}".format(batch))
#                     sys.stdout.flush()

        except:
            if DEBUG: print("No batches of data ,",batches)
            if countSum==0: break
            costVal=costCountProductSum/countSum
            accVal=trainOrTest,accCountProductSum/countSum
            print("{} accuracy= {}, cost= {}".format(trainOrTest,accVal,costVal))
            return accVal,costVal
            break


# In[18]:


FRAME_HEIGHT=100
FRAME_WIDTH=200
COLOR_CHANNELS=3
CELLS_PER_FRAME=25

INPUT_DIM=(FRAME_HEIGHT,FRAME_WIDTH)
OUTPUT_DIM=CELLS_PER_FRAME


EPOCHS=10
BATCH_SIZE=64
CUDA1=0
CUDA2=1

os.environ["CUDA_VISIBLE_DEVICES"]="{},{}".format(CUDA1,CUDA2)
sess = tf.Session()


# In[19]:


'''
BLACK AND WHITE VIDEO

'''
FILE_NAME='./video/bw'
FILE_NAME_VIDEO=FILE_NAME+'.avi'
FILE_NAME_CSV=FILE_NAME+'.csv'

NO_FRAMES=10000
if PC: NO_FRAMES=100
FRAMES_TO_SKIP=0

dataX=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
dataY=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)




if DEBUG: print("X type: {}, Y type: {}.".format(dataX.dtype,dataY.dtype))

xTrain, xTest, yTrain, yTest= train_test_split(dataX, dataY, test_size=0.1)

print("SIZES: xTrain {}, yTrain {}, xTest {}, yTest {}".format(xTrain.shape,yTrain.shape,xTest.shape,yTest.shape))


placeX=tf.placeholder(tf.float32,shape=[None,INPUT_DIM[0],INPUT_DIM[1],COLOR_CHANNELS])
placeY=tf.placeholder(tf.float32,shape=[None,CELLS_PER_FRAME])

data=(placeX,placeY)
dataset=tf.data.Dataset.from_tensor_slices(data)
#dataset=dataset.shuffle(buffer_size=NO_FRAMES,reshuffle_each_iteration=True)
dataset=dataset.batch(BATCH_SIZE)
iterr=dataset.make_initializable_iterator()




# In[20]:


'''xColour=tf.placeholder(tf.float32,shape=[None,INPUT_DIM[0],INPUT_DIM[1],COLOR_CHANNELS],name='xColour')
yTrue=tf.placeholder(tf.float32,shape=[None,CELLS_PER_FRAME],name='yTrue')'''

xColour,yTrue=iterr.get_next()
xGrey=tf.reduce_mean(xColour-128.0,axis=-1)/255.0
if DEBUG: print("DEBUG: Shapes xColour {} yTure {} xGrey {}".format(xColour.shape,yTrue.shape,xGrey.shape))

l1=inputLayer(xGrey,[INPUT_DIM[0],INPUT_DIM[1]])
l2=flatten(l1)
l3=fullyConnectedLayer(l2,OUTPUT_DIM)
l4=sigmoid(l3)
yPred,opt,acc,cst=buildNetwork(l4,yTrue)


sess.run(tf.global_variables_initializer())


if DEBUG: print("layer shapes 1:{} 2:{} 3:{} ".format(l1.shape,l2.shape,l3.shape))
    


# In[24]:


for i in range(EPOCHS):
    trainAcc,trainCost=printAccuracy(iterr,xTrain,yTrain,acc,cst,"Training")
    testAcc,testCost=printAccuracy(iterr,xTest,yTest,acc,cst,"Testing")
    if min(trainCost,testCost) < TARGET_COST:
        print("Converged!")
        break
        
    train(iterr,opt,1,xTrain,yTrain)


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


dataX=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
dataY=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)




if DEBUG: print("X type: {}, Y type: {}.".format(dataX.dtype,dataY.dtype))

xTrain, xTest, yTrain, yTest= train_test_split(dataX, dataY, test_size=0.1)

print("SIZES: xTrain {}, yTrain {}, xTest {}, yTest {}".format(xTrain.shape,yTrain.shape,xTest.shape,yTest.shape))



# In[ ]:


for i in range(EPOCHS):
    trainAcc,trainCost=printAccuracy(iterr,xTrain,yTrain,acc,cst,"Training")
    testAcc,testCost=printAccuracy(iterr,xTest,yTest,acc,cst,"Testing")
    if min(trainCost,testCost) < TARGET_COST:
        print("Converged!")
        break
        
    train(iterr,opt,1,xTrain,yTrain)


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


dataX=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
dataY=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)




if DEBUG: print("X type: {}, Y type: {}.".format(dataX.dtype,dataY.dtype))
xTrain, xTest, yTrain, yTest= train_test_split(dataX, dataY, test_size=0.1)
print("SIZES: xTrain {}, yTrain {}, xTest {}, yTest {}".format(xTrain.shape,yTrain.shape,xTest.shape,yTest.shape))



# In[ ]:


for i in range(EPOCHS):
    trainAcc,trainCost=printAccuracy(iterr,xTrain,yTrain,acc,cst,"Training")
    testAcc,testCost=printAccuracy(iterr,xTest,yTest,acc,cst,"Testing")
    if min(trainCost,testCost) < TARGET_COST:
        print("Converged!")
        break
        
    train(iterr,opt,1,xTrain,yTrain)


# In[ ]:





# In[ ]:




