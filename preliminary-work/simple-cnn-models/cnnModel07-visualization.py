#!/usr/bin/env python
# coding: utf-8

# In[19]:


DEBUG=False

FRAME_WIDTH=200
FRAME_HEIGHT=100


# In[35]:


def displayWeights(W,outputNode):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X=np.arange(FRAME_WIDTH)
    Y=np.arange(FRAME_HEIGHT)
    X,Y=np.meshgrid(X, Y)
    Z=W

    if DEBUG: print(Z.shape)
    Z=np.reshape(Z[:,outputNode],(FRAME_HEIGHT,FRAME_WIDTH))



    if DEBUG: print(X.shape,Y.shape,Z.shape)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

    ax.set_zlim(np.min(Z), np.max(Z))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.title("NN weigts deciding the cell no. {}".format(outputNode))
    plt.xlabel('Screen width')
    plt.ylabel('Screen height')
    ax.set_zlabel('\n' + 'NN weights\n')



    # Add a color bar which maps values to colors.
    fig.colorbar(surf)#, shrink=0.5, aspect=5)
    plt.show()


# In[36]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


data=np.load('./weights/cnnModel07-weights.npz')


w3Ran=data['w3Ran']

data=np.load('./weights/cnnModel07-weights-bwoverfit.npz')
w3Bw=data['w3Bw']

data=np.load('./weights/cnnModel07-weights-multiplevideos-20.npz')
w3MultipleVideo20=data['w3MultipleVideo']

data=np.load('./weights/cnnModel07-weights-multiplevideos-50.npz')
w3MultipleVideo50=data['w3MultipleVideo']


for o in range(25):
    displayWeights(w3MultipleVideo50,o)


# In[ ]:




