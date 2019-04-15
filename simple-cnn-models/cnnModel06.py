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
NO_MAX_FRAMES=10000



TARGET_COST=0.00001


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


def maxPoolingLayer(inputTensor,kernelSize):
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
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    yPred=tf.round(outputTensor)
    if DEBUG: print("DEBUG: yPred shape= {}".format(yPred.shape))
    correctPrediction=tf.equal(yPred,targetTensor)
    if DEBUG: print("DEBUG: correctPrediction shape= {}".format(correctPrediction.shape))
    accuracy=tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
    if DEBUG: print("DEBUG: accuracy shape= {}".format(accuracy.shape))
    
    
    print("yPred",yPred.shape,"Correct pred",correctPrediction.shape,"Accuracy",accuracy.shape,"Sq error",sqError.shape,"Cost",cost.shape)    
    
    if DEBUG: print("DEBUG: Completed building network")
    return outputTensor,optimizer,accuracy,cost


# In[13]:


def train(iterr,optimizer,noIterations,X,Y):
    DISPLAY_PROGRESS_EVERY_ITER=1
    DISPLAY_PROGRESS_EVERY_BATCH=25
    
    global currentTotalIterations
    
    
    for it in range(currentTotalIterations,currentTotalIterations+noIterations):
        sess.run(iterr.initializer,feed_dict={placeX:X,placeY:Y})
        
        batch=0
        while True:
            try:   
                sess.run(optimizer)
                batch+=1
            except:
                print("\t {} batches trained".format(batch))
                break 
        
        if (it>0)  and (it%DISPLAY_PROGRESS_EVERY_ITER==0):
            print("\t Iter {} complete".format(it+1))
    currentTotalIterations+=noIterations
    print("\tDEBUG: train() function completed")


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
        
    for f in range(noFrames):
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
            countSum+=1
            batches+=1

        except:
            if DEBUG: print("\t No batches of data ".format(batches))
            if countSum==0: break
            costVal=costCountProductSum/countSum
            accVal=trainOrTest,accCountProductSum/countSum
            print("\t {} accuracy= {}, cost= {}".format(trainOrTest,accVal,costVal))
            return accVal,costVal
            break


# In[18]:


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
sess = tf.Session()


# In[19]:


def trainAndTestForVideo(fileName,noFrames,framesToSkip=0,videoFileFormat='.avi',testSplit=0.1):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Starting new video file\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}".format(fileName))
    
    FILE_NAME=fileName
    FILE_NAME_VIDEO=FILE_NAME+videoFileFormat
    FILE_NAME_CSV=FILE_NAME+'.csv'

    NO_FRAMES=noFrames
    if PC: NO_FRAMES=100
    FRAMES_TO_SKIP=framesToSkip

    dataX=makeX(FILE_NAME_VIDEO,NO_FRAMES,FRAMES_TO_SKIP)
    dataY=makeY(FILE_NAME_CSV,NO_FRAMES,FRAMES_TO_SKIP)
    
    
    
    if DEBUG: print("X type: {}, Y type: {}.".format(dataX.dtype,dataY.dtype))
    xTrain, xTest, yTrain, yTest= train_test_split(dataX, dataY, test_size=testSplit)
    print("SIZES: xTrain {}, yTrain {}, xTest {}, yTest {}".format(xTrain.shape,yTrain.shape,xTest.shape,yTest.shape))
    
    hist={"trainAcc":[],"trainCost":[],"testAcc":[],"testCost":[]}
    
    
    for i in range(EPOCHS):
        trainAcc,trainCost=printAccuracy(iterr,xTrain,yTrain,acc,cst,"Training")
        testAcc,testCost=printAccuracy(iterr,xTest,yTest,acc,cst,"Testing")
        
        hist["trainAcc"].append(trainAcc)
        hist["trainCost"].append(trainCost)
        hist["testAcc"].append(testAcc)
        hist["testCost"].append(testCost)
        
        
        if min(trainCost,testCost) < TARGET_COST:
            print("Converged!")
            break

        train(iterr,opt,1,xTrain,yTrain)
    
    return hist
    


# In[20]:


placeX=tf.placeholder(tf.float32,shape=[None,INPUT_DIM[0],INPUT_DIM[1],COLOR_CHANNELS])
placeY=tf.placeholder(tf.float32,shape=[None,CELLS_PER_FRAME])

data=(placeX,placeY)
dataset=tf.data.Dataset.from_tensor_slices(data)
dataset=dataset.shuffle(buffer_size=NO_MAX_FRAMES,reshuffle_each_iteration=True)
dataset=dataset.batch(BATCH_SIZE)
iterr=dataset.make_initializable_iterator()




# In[21]:


'''xColour=tf.placeholder(tf.float32,shape=[None,INPUT_DIM[0],INPUT_DIM[1],COLOR_CHANNELS],name='xColour')
yTrue=tf.placeholder(tf.float32,shape=[None,CELLS_PER_FRAME],name='yTrue')'''

xColour,yTrue=iterr.get_next()
xGrey=tf.reduce_mean(xColour,axis=-1)
xGrey=(xGrey-128.0)/256.0
if DEBUG: print("DEBUG: Shapes xColour {} yTure {} xGrey {}".format(xColour.shape,yTrue.shape,xGrey.shape))

l1=inputLayer(xGrey,[INPUT_DIM[0],INPUT_DIM[1]])
l2=flatten(l1)
l3=fullyConnectedLayer(l2,OUTPUT_DIM)
l4=sigmoid(l3)
yPred,opt,acc,cst=buildNetwork(l4,yTrue)


sess.run(tf.global_variables_initializer())


if DEBUG: print("layer shapes 1:{} 2:{} 3:{} ".format(l1.shape,l2.shape,l3.shape))
    


# In[22]:


histBw=trainAndTestForVideo('./video/bw',10000)


# In[23]:


histRan=trainAndTestForVideo('./video/ran',100000)


# In[24]:


histMultipleVideo=trainAndTestForVideo('./video/multipleVideos',10000,videoFileFormat='.mp4',testSplit=0.5)


# In[25]:


print(histBw)
print(histRan)
print(histMultipleVideo)


# In[ ]:


'''
{'trainCost': [0.487190250597947, 6.275400916450763e-06], 
'testAcc': [('Testing', 0.49980468675494194), ('Testing', 1.0)], 
'testCost': [0.48725608363747597, 1.0347399794592357e-07], 
'trainAcc': [('Training', 0.4994202127270665), ('Training', 0.9999911345488636)]}


{'trainCost': [0.20937026709529524, 0.061433447273910466, 0.026764632209290005, 0.015015229714889052, 0.009539862714892796, 0.006624541298241903, 0.004842561076837757, 0.003697494075448642, 0.0029006636097155354, 0.0023166576849547684, 0.0019074803687725381, 0.0015896577150676505, 0.0013644386110030584, 0.001186302326459554, 0.0010262928389373825, 0.0008936834280917434, 0.0007987172024358903, 0.0007147448452139304, 0.0006410042675296034, 0.0005829130136686566], 
'testAcc': [('Testing', 0.7569765672087669), ('Testing', 0.9179531298577785), ('Testing', 0.9588046856224537), ('Testing', 0.9735781326889992), ('Testing', 0.9810937456786633), ('Testing', 0.9852421917021275), ('Testing', 0.9873281307518482), ('Testing', 0.9882656261324883), ('Testing', 0.9894609339535236), ('Testing', 0.9904375039041042), ('Testing', 0.9911250062286854), ('Testing', 0.9915624968707561), ('Testing', 0.9914609342813492), ('Testing', 0.9920546822249889), ('Testing', 0.9925390593707561), ('Testing', 0.9926249943673611), ('Testing', 0.9929453171789646), ('Testing', 0.9930312484502792), ('Testing', 0.9930156245827675), ('Testing', 0.9933359436690807)], 
'testCost': [0.20510075148195028, 0.06685484061017632, 0.03253534168470651, 0.020639818103518337, 0.01502221537521109, 0.011874609510414302, 0.010066515096696094, 0.009086779988138005, 0.008293081540614367, 0.007626592996530235, 0.007099631344317459, 0.006742824451066554, 0.006500241201138124, 0.006219661634531803, 0.005867137078894302, 0.005834336916450411, 0.005574006267124787, 0.00550500194367487, 0.0053980302036507055, 0.0052129948599031195], 
'trainAcc': [('Training', 0.7515354613040356), ('Training', 0.9251923764005621), ('Training', 0.9677101048171943), ('Training', 0.9822925510981404), ('Training', 0.9889725218427942), ('Training', 0.9924822703320929), ('Training', 0.9946498211393965), ('Training', 0.9959246419000287), ('Training', 0.9968235797070443), ('Training', 0.9975150671411068), ('Training', 0.9979884721708636), ('Training', 0.9983262360518705), ('Training', 0.9985983929735549), ('Training', 0.9987508764503695), ('Training', 0.9989361602363857), ('Training', 0.9990868716375202), ('Training', 0.9991790673411485), ('Training', 0.9992748122688726), ('Training', 0.999352827985236), ('Training', 0.9994015862755742)]}


{'trainCost': [0.5029900013645993, 0.4776768835285042, 0.4591610099695906, 0.44652620636964147, 0.42667750467227983, 0.4050721575187731, 0.38265376641780513, 0.3606759505935862, 0.3475723809833768, 0.32737633812276623, 0.3119316508498373, 0.2990805570837818, 0.2864646523059169, 0.2791970398607133, 0.2705821962673453, 0.26238762823086753, 0.2565659776895861, 0.24874630155442637, 0.2458962330335303, 0.2412038983046254], 
'testAcc': [('Testing', 0.4862816446944128), ('Testing', 0.5078481017034265), ('Testing', 0.5216139245636856), ('Testing', 0.5325632868688318), ('Testing', 0.5485759513287605), ('Testing', 0.5647943042501619), ('Testing', 0.5853639260123048), ('Testing', 0.6034256348127052), ('Testing', 0.6154746858379508), ('Testing', 0.6347151873986933), ('Testing', 0.6476028493688076), ('Testing', 0.6608702540397644), ('Testing', 0.67348101320146), ('Testing', 0.680324366575555), ('Testing', 0.6890348075311395), ('Testing', 0.6955775286577925), ('Testing', 0.70223892565015), ('Testing', 0.7094066444831558), ('Testing', 0.7109651889982103), ('Testing', 0.7164477856853341)], 
'testCost': [0.5040931848785545, 0.4840255138240283, 0.4693682189983658, 0.4575264374666576, 0.44012309667430344, 0.4228188406817521, 0.4004590971560418, 0.382104583556139, 0.3697694204276121, 0.35061238234556175, 0.3376556023766723, 0.3246032493778422, 0.3124394220641897, 0.3059287022186231, 0.2977016475758975, 0.2912550908100756, 0.2847993453092213, 0.2783400008950052, 0.2768173849658121, 0.2716038244057305], 
'trainAcc': [('Training', 0.48648733919179893), ('Training', 0.5144857590711569), ('Training', 0.532064874715443), ('Training', 0.5438528490971916), ('Training', 0.5622626588314394), ('Training', 0.5832911382747602), ('Training', 0.6044620275497437), ('Training', 0.626368674296367), ('Training', 0.6394382926482188), ('Training', 0.6598180434371852), ('Training', 0.6762341781507565), ('Training', 0.6898813262770448), ('Training', 0.7035522151596939), ('Training', 0.7105221499370623), ('Training', 0.7195411384860172), ('Training', 0.7288370238074774), ('Training', 0.7352531650398351), ('Training', 0.7440980997266649), ('Training', 0.7474762646457817), ('Training', 0.7525158207627791)]}
'''

