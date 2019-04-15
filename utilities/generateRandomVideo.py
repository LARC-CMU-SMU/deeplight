#!/usr/bin/env python
# coding: utf-8

# In[ ]:


MIN=50
MAX=200
OFFSET=20


# In[4]:


DEBUG=True
PC=False


# In[5]:


import numpy as np
import cv2
import pandas as pd 


# In[6]:


NO_FRAMES=10000
if PC: NO_FRAMES=100
FRAME_HEIGHT=100
FRAME_WIDTH=200
COLOR_CHANNELS=3

CELLS_VERTICLE=5
CELLS_HORIZONTAL=5
CELLS_PER_FRAME=CELLS_VERTICLE*CELLS_HORIZONTAL


CELL_HEIGHT=int(FRAME_HEIGHT/CELLS_VERTICLE)
CELL_WIDTH=int(FRAME_WIDTH/CELLS_HORIZONTAL)

FILE_NAME='./video/ran'
FILE_NAME_VIDEO=FILE_NAME+'.avi'
FILE_NAME_CSV=FILE_NAME+'.csv'


fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(FILE_NAME_VIDEO, fourcc, 20, (FRAME_WIDTH,FRAME_HEIGHT))

csvToSave=np.zeros((NO_FRAMES,CELLS_PER_FRAME),dtype=np.int32)

def cellIndexToYX(idx):
    x=int(idx%CELLS_HORIZONTAL)
    y=int(idx/CELLS_HORIZONTAL)
    return y,x


for f in range(NO_FRAMES):
    msg=np.random.random_integers(0,1,CELLS_PER_FRAME)
    
    
    frame=np.random.random_integers(low=MIN, high=MAX, size=(FRAME_HEIGHT,FRAME_WIDTH,COLOR_CHANNELS))
    frame=frame.astype(np.uint8)
    
    
    for cell in range(CELLS_PER_FRAME):
        yIdx,xIdx=cellIndexToYX(cell)
        
        if msg[cell]==1:
            #if DEBUG: print("yIdx,xIdx",yIdx,xIdx)
            frame[yIdx*CELL_HEIGHT:(yIdx+1)*CELL_HEIGHT,xIdx*CELL_WIDTH:(xIdx+1)*CELL_WIDTH,:]+=OFFSET
        else:
            frame[yIdx*CELL_HEIGHT:(yIdx+1)*CELL_HEIGHT,xIdx*CELL_WIDTH:(xIdx+1)*CELL_WIDTH,:]-=OFFSET
        
    
    csvToSave[f,:]=msg[:]
    
    
    video.write(frame)
    if PC:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    
    
    if DEBUG: print("Frame {} complete".format(f))
    
video.release()
cv2.destroyAllWindows()
np.savetxt(FILE_NAME_CSV, csvToSave, delimiter=",",fmt='%d')

if DEBUG: print("Complete")


# In[ ]:




