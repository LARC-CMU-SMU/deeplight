import numpy as np
import cv2 as cv
import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="6"
from sklearn.model_selection import train_test_split
from skimage.transform import rotate

AUGMENTATION_FACTOR=3
AUGMENTATION_FACTOR_2=2

NN_H=64
NN_W=64

X=[]
Y=[]

def writeImageSet(X,offset=0,scale=255,savePrefix="output_"):
	X=((X+offset)*scale)
	# X=np.clip(X,0,255)
	print("Range of images",np.min(X),np.max(X))
	X=X.astype(np.uint8)


	for i in range(len(X)):
		cv.imwrite("{}{}.jpg".format(savePrefix,i),X[i])

def greyscaletoRGB(X):
	return np.concatenate((X,X,X),axis=-1)

def unet(pretrained_weights=None, input_size=(NN_H, NN_W, 3)):
	inputs = Input(input_size)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(drop5))
	merge6 = concatenate([drop4, up6], axis=3)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv6))
	merge7 = concatenate([conv3, up7], axis=3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv7))
	merge8 = concatenate([conv2, up8], axis=3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv8))
	merge9 = concatenate([conv1, up9], axis=3)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

	model = Model(input=inputs, output=conv10)

	model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

	# model.summary()

	if (pretrained_weights):
		model.load_weights(pretrained_weights)

	return model


def dottedLine(img,pt1,pt2,color,thickness=3):
	gap=thickness*3
	dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
	pts= []
	for i in  np.arange(0,dist,gap):
		r=i/dist
		x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
		y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
		p = (x,y)
		pts.append(p)

	for p in pts:
		cv.circle(img,p,thickness,color,-1)


def markImg(imgIn,cor,col=[255,255,255],thicc=3,dotted=False):
	img = None
	if len(imgIn.shape)==2:
		imgIn=np.reshape(imgIn,(imgIn.shape[0],imgIn.shape[1],1))
	if imgIn.shape[2]==1:
		# print("AAA")
		imgIn=np.concatenate((imgIn,imgIn,imgIn),axis=2)


	img=np.array(imgIn,dtype=np.uint8)
	print(img.shape)
	# input("Okau?")
	for i in range(4):
		cv.circle(img,center=(cor[i][0],cor[i][1]),color=col,radius=2,thickness=thicc)
		if not dotted:
			cv.line(img, (cor[i][0],cor[i][1]),(cor[(i+1)%4][0],cor[(i+1)%4][1]), color=col, thickness=thicc)
		else:
			dottedLine(img, (cor[i][0],cor[i][1]),(cor[(i+1)%4][0],cor[(i+1)%4][1]), color=col, thickness=thicc)

	return img



if __name__=='__main__':

	args=argparse.ArgumentParser()
	args.add_argument("--epochs","-ep", dest="epochs", type=int)
	args.add_argument("--loadWeights","-lw",dest="loadWeights",type=str)
	args.add_argument("--saveWeights", "-sw", dest="saveWeights", type=str)
	args.add_argument("--outputFolder", "-o", dest="outputFolder", type=str)
	args.add_argument("--iou",dest="iou", type=bool)


	args=args.parse_args()

	
	for x in range(0,400+1):
	# for x in range(120,120+8+1):
		try:
			imgRaw=cv.imread("frames/frame_{}.jpeg".format(x))
			npyy=np.load("frames/frame_{}.npy".format(x))
			img=cv.resize(imgRaw,(NN_W,NN_H))

			args_inHeight=imgRaw.shape[0]
			args_inWidth=imgRaw.shape[1]

			args_outHeight=NN_H
			args_outWidth=NN_W
			npyy[:,1]=((1.0*npyy[:,1]*args_outHeight)/(1.0*args_inHeight)).astype(npyy.dtype)
			npyy[:,0]=((1.0*npyy[:,0]*args_outWidth)/(1.0*args_inWidth)).astype(npyy.dtype)
			# print("AAA")
			X.append(img)
			Y.append(npyy)
		except:
			print("Error reading SMU : {}. Current total {}".format(x,len(X)))


	for x in range(0,368+1):
	# for x in range(0,8+1):
		try:
			imgRaw=cv.imread("refined/{}.jpg".format(x))
			npyy=np.load("refined/{}.npy".format(x))
			img=cv.resize(imgRaw,(NN_W,NN_H))

			args_inHeight=imgRaw.shape[0]
			args_inWidth=imgRaw.shape[1]

			args_outHeight=NN_H
			args_outWidth=NN_W
			npyy[:,1]=((1.0*npyy[:,1]*args_outHeight)/(1.0*args_inHeight)).astype(npyy.dtype)
			npyy[:,0]=((1.0*npyy[:,0]*args_outWidth)/(1.0*args_inWidth)).astype(npyy.dtype)
			# print("AAA")
			X.append(img)
			Y.append(npyy)
		except:
			print("Error reading Refined : {}. Current total {}".format(x,len(X)))




	X=np.array(X)
	Y=np.array(Y)



	print("1. PROPERTIES: X, Y")
	print(np.min(X),np.mean(X),np.max(X))
	print(np.min(Y),np.mean(Y),np.max(Y))



	'''Generate Y images'''
	YY=[]
	for i in range(len(X)):
		outImg=np.zeros((NN_H,NN_W,3),dtype=np.uint8)
		cv.fillConvexPoly(outImg,points=Y[i,:,:],color=(255,255,255))
		outImg=cv.cvtColor(outImg,code=cv.COLOR_BGR2GRAY)
		YY.append(outImg)
	Y = np.array(YY, dtype=np.float32) / 255.0  # Clear the memeory for Y
	Y = np.resize(Y, (len(Y), NN_H, NN_W, 1))
	YY = None



	print("2. PROPERTIES: X, Y")
	print(np.min(X),np.mean(X),np.max(X))
	print(np.min(Y),np.mean(Y),np.max(Y))




	import random
	XX=[]
	YY=[]


	if (args.outputFolder==None):
		for i in range(X.shape[0]):
			for j in range(AUGMENTATION_FACTOR):
				for k in range(AUGMENTATION_FACTOR_2):
					leftTopY=random.randint(0,int(NN_H/5))
					leftTopX=random.randint(0,int(NN_W/5))
					rightBottomY=random.randint(int((4*NN_H)/5),NN_H)
					rightBottomX=random.randint(int((4*NN_W)/5),NN_W)

					X2=X[i,leftTopY:rightBottomY,leftTopX:rightBottomX,:]
					Y2=Y[i,leftTopY:rightBottomY,leftTopX:rightBottomX,:]
					X2=cv.resize(X2,(NN_W,NN_H))
					Y2=cv.resize(Y2,(NN_W,NN_H))


					# print("X2",X2.shape)
					# print("Y2",Y2.shape)

					ang=(random.random()-0.5)*90


					XX.append(rotate(X2,angle=ang,mode='edge'))
					YY.append(rotate(Y2,angle=ang,mode='constant'))

		X=np.array(XX)
		Y=np.array(YY)
		Y = np.resize(Y, (len(Y), NN_H, NN_W, 1))

		XX=None
		YY=None
	else:
		X=X/255.0

	print("3. PROPERTIES: X, Y")
	print(np.min(X),np.mean(X),np.max(X))
	print(np.min(Y),np.mean(Y),np.max(Y))

	X= 2.0*X - 1.0
	Y= Y

	print("4. PROPERTIES: X, Y")
	print(np.min(X),np.mean(X),np.max(X))
	print(np.min(Y),np.mean(Y),np.max(Y))



	if args.outputFolder==None:
		XY=np.concatenate( (X , greyscaletoRGB(Y*2.0 - 1.0)) , axis=1)
		writeImageSet(XY,offset=1.0,scale=127.0,savePrefix="temp/aa")



	# if args.loadWeights==None:
	if True:
		xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.01, random_state=1)
	else:
		xTest=X


	# print(xTrain[0])
	# print(yTrain[0])
	print("5.PROPERTIES: xTrain, yTrain")
	print(np.min(xTrain),np.mean(xTrain),np.max(xTrain))
	print(np.min(yTrain),np.mean(yTrain),np.max(yTrain))


	_ = str(input("Shall I proceed?"))

	model=unet(pretrained_weights=None,input_size=(NN_H,NN_W,3))

	if (args.loadWeights!=None):
		model.load_weights(args.loadWeights)
		print("Loaded weights from {}".format(args.loadWeights))
	if (args.epochs!=None):
		model.fit(xTrain,yTrain,epochs=args.epochs)
	if (args.saveWeights!=None):
		model.save_weights(args.saveWeights)
		print("Weights saved to {}".format(args.saveWeights))
	if (args.outputFolder!=None):
		yPredAr=model.predict(X)
	if (args.iou!=None):
		if (args.iou==True):
			yPredAr=(yPred*255).astype(np.uint8)
			
			yPredCol=np.concatenate((yPred,yPred,yPred),axis=2)
			yPred=cv.cvtColor(yPred,cv.COLOR_BGR2GRAY)




			ret, thresh = cv.threshold(yPred,60,255,cv.THRESH_BINARY)
			kernel = np.ones((2, 2), np.uint8)
			dilated = cv.dilate(thresh, kernel, iterations=3)


			contours, hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			dilated=np.reshape(dilated,(256,256,1))
			yPred=np.concatenate((dilated,dilated,dilated),axis=2).astype(np.uint8)


			contours, hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			contours=sorted(contours, key=cv.contourArea)
			bestContour=contours[-1]

			M = cv.moments(bestContour)
			xMid = int(M["m10"] / M["m00"])
			yMid = int(M["m01"] / M["m00"])
			m=np.reshape(bestContour,(bestContour.shape[0],2))

			temp = np.square(m[:, 0] - xMid) + np.square(m[:, 1] - yMid)
			lt = np.argmax((m[:, 0] < xMid).astype(np.float32) * (m[:, 1] < yMid).astype(np.float32) * temp)
			rt = np.argmax((m[:, 0] > xMid).astype(np.float32) * (m[:, 1] < yMid).astype(np.float32) * temp)
			rd = np.argmax((m[:, 0] > xMid).astype(np.float32) * (m[:, 1] > yMid).astype(np.float32) * temp)
			ld = np.argmax((m[:, 0] < xMid).astype(np.float32) * (m[:, 1] > yMid).astype(np.float32) * temp)

			scrCorners=np.array([m[lt], m[rt], m[rd], m[ld]])

			scrCorners[:,0]=scrCorners[:,0]*(img.shape[1]/255.0)
			scrCorners[:,1]=scrCorners[:,1]*(img.shape[0]/255.0)

			BLACK=[0,0,0]
			WHITE=[255,255,255]
			GREEN=[0,255,0]







		XY=np.concatenate( (X , greyscaletoRGB(Y*2.0 - 1.0),greyscaletoRGB(yPred*2.0 - 1.0)) , axis=1)
		writeImageSet(XY,offset=1.0,scale=127.0,savePrefix=args.outputFolder)



		

	print(X.shape)
	print(Y.shape)
	
	print("END OF PROGRAM")


