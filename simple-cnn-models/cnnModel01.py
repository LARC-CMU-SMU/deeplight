import numpy as np
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Conv2D,Input,Dropout
import sys

import os


def rnn(inptuDim,outputDim):
    model=Sequential()
    model.add(Conv2D(5, kernel_size=(5,5), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(5,5), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Conv2D(5, kernel_size=(2,2), activation='relu',strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(10000,activation='relu'))
    model.add(Dense(2000,activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense((OUTPUT_DIM),activation='sigmoid'))

    return model

print("python fineName EPOCHS BATCH_SIZE CUDA1 CUDA2 TypeSomethingIfFirstRun")

INPUT_DIM=(0,0,0)
OUTPUT_DIM=0
EPOCHS=int(sys.argv[1])
BATCH_SIZE=int(sys.argv[2])
CUDA1=int(sys.argv[3])
CUDA2=int(sys.argv[4])


os.environ["CUDA_VISIBLE_DEVICES"]="{},{}".format(CUDA1,CUDA2)


X=np.load("X.npy")/255.0
Y=np.load("Y.npy")

X=np.mean(X,axis=3)
X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))


print(np.shape(X),np.shape(Y))
xTrain, xTest, yTrain, yTest= train_test_split(X, Y, test_size=0.33)


INPUT_DIM=X[0].shape
OUTPUT_DIM=int((Y[0]).shape[0])

print("X size: {}, Y size: {}".format(np.shape(xTrain),np.shape(yTrain)))
print("Input dim: ",INPUT_DIM," . ","Output dim: ",OUTPUT_DIM)


model=rnn(INPUT_DIM,OUTPUT_DIM)
model.build()
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
if(len(sys.argv)==5):
    print(model.summary())
model.fit(xTrain,yTrain,validation_split=0.3,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE,shuffle=False)


print(model.evaluate(xTest,yTest))
'''model.fit(xTrain, yTrain, validation_split=0.2, shuffle=True,epochs=10)
scores = model.evaluate(xTest, yTest)'''





