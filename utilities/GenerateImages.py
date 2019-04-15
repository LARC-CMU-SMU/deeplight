import numpy as np
import sys
import cv2

data=np.load("data_256_5_mix.npz")
X=data['arr_0']
Y=data['arr_0']

#idx=int(sys.argv[1])


for i in range(0,1000):
    cv2.imwrite('./pics/frame{}.jpg'.format(i),X[i,:,:,:])


#print(Y[idx,:])
