#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import sys

data=np.load('../video/butterfly.npz')
X=data['X']
Y=data['Y']


print("DEBUG: ","X",X.shape,"Y",Y.shape)

N=int(sys.argv[1])

print(Y[N])
cv2.imshow("Image",X[N])
cv2.waitKey(100000000)



for i in range(X.shape[0]):
    #idx=int(input())
    #cv2.ion()
    cv2.imshow("Image",X[i])
    cv2.waitKey(10)
    print(i,Y[i])
    
cv2.destroyAllWindows()

    


# In[ ]:





# In[ ]:




