# Deeplight : Robust & Unobtrusive Real-time Screen-Camera Communication for Real-World Displays

This repository is the authors' implementation of the following paper.

Vu Tran, Gihan Jayatilaka, Ashwin Ashok and Archan Misra, 2021, April. *Deeplight : Robust & Unobtrusive Real-time Screen-Camera Communication for Real-World Displays*. In 2021 20th ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN) (). IEEE.

## Overview diagram
![Deeplight overview](./documentation/overview.png)

## System architecture
![System architecture](./documentation/system-architecture.png)

DeepLight's objective is to enable Screen-Camera Communication in real-world conditions where the camera is hand-held, and the modulation must be imperceptible at COMMON display rates (30-60FPS). In these conditions, the state-of-the-art is susceptible to screen extraction error (losing part of the screen or including the content close to the screen) as the decoder explicitly split the extracted screen into a grid. Consequently, a slight deviation from the original grid may result in a large percentage of a cell being translated into neighboring cells and vice versa. In addition, the state-of-the-art relies on high frequencies to suppress the flickers created by the modulation. DeepLight is robust against screen extraction error as it does not explicitly split the extracted screen into a grid. Instead, it decodes each bit using the information of the entire screen via a convolutional neural network. Thank to the holistic decoding approach, DeepLight is sufficiently robust to support high decoding accuracy even with a hand-held camera (with hand-motion artifacts). Of course, DeepLight still needs to localize and extract the screen from camera frames with reasonably high accuracy. DeepLight applies a deep-learning-based, pixel-wise segmentation method to localize the screen. It then uses a RANSAC-like algorithm with contour analysis to detect the border of the screen with a practically high IOU of 89% in indoor environments. Finally, DeepLight achieves a significantly higher Mean Opinion Score (MOS) compared with the state-of-the-art by selectively encoding the data in only the Blue channel (primate eyes are known to be less sensitive to Blue color due to lower number of receptors). 

In this project, we implement DeepLight in both offline (using python) and real-time (using Objective-C for iPhone) modes. For the offline mode, we use Keras and Tensorflow to implement the neural network. For the real-time mode, we port the code into Objective-C and C++. We use the CoreML framework to execute the neural networks and use the OpenCV framework for typical image processing tasks.

## Code structure


# Cite

You may cite this work as per the following bibitex.
```
@InProceedings{Tran2021IPSN,
author = {Tran, Vu and Jayatilaka, Gihan and Ashok, Ashwin and Misra, Archan},
title = {Deeplight : Robust & Unobtrusive Real-time Screen-Camera Communication for Real-World Displays},
booktitle = {ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN)},
month = {May},
year = {2021}
}
```