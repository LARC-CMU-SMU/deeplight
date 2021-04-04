import tensorflow as tf
# tf.set_random_seed(625742)
from numpy.random import seed
seed(625742)
# from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import multi_gpu_model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image
import numpy as np

CELLS_PER_FRAME = 100

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)

    return x

def deeplightmodel():
    imgInput = Input(shape=(299,299,3), name="input")
    x = conv2d_bn(imgInput, 16, 1, 1, padding='valid', name="1st")         #299
    # x = conv2d_bn(x, 64, 1, 1, padding='valid')                 #299
    x = conv2d_bn(x, 32, 9, 9, strides=(3, 3), padding='valid', name="2nd")  #
    x = conv2d_bn(x, 64, 7, 7, strides=(3, 3), padding='valid', name="3rd")
    x = conv2d_bn(x, 100, 5, 5, strides=(2, 2), padding='valid', name="4th")
    x = conv2d_bn(x, 100, 3, 3, strides=(1, 1), padding='valid', name="5th")
    x = Dropout(0.25)(x)
    x = Conv2D(CELLS_PER_FRAME, (12, 12), padding='valid')(x)
    x = Activation('sigmoid')(x)
    x = Flatten()(x)
    model = Model(imgInput, x, name='deeplight')
    return model

if __name__ == '__main__':
    model = deeplightmodel()
    print(model.summary())
