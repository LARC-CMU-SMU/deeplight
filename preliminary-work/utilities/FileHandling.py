import numpy as np
import sys


print("python fileName dataFileName noOfFrames XsaveName YsaveName")

file=np.load(sys.argv[1])
X=file["arr_0"]
Y=file["arr_1"]

np.save(sys.argv[3],X[0:int(sys.argv[2]),:,:,:])
np.save(sys.argv[4],Y[0:int(sys.argv[2]),:])


print('Done')
