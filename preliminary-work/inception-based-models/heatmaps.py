import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras import backend as K

from heatmap import to_heatmap, synset_to_dfs_ids
from keras.models import load_model

def display_heatmap(new_model, img_path, ids, preprocessing=None):
    # The quality is reduced.
    # If you have more than 8GB of RAM, you can try to increase it.
    img = image.load_img(img_path, target_size=(800, 1280))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if preprocessing is not None:
        x = preprocess_input(x)

    out = new_model.predict(x)

    heatmap = out[0]  # Removing batch axis.

    if K.image_data_format() == 'channels_first':
        heatmap = heatmap[ids]
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=0)
    else:
        heatmap = heatmap[:, :, ids]
        if heatmap.ndim == 3:
            heatmap = np.sum(heatmap, axis=2)

    plt.imshow(heatmap, interpolation="none")
    plt.show()


model = load_model('InceptionV3.h5')
new_model = to_heatmap(model)
idx = 0  # The index of the class you care about, here the first one.
display_heatmap(new_model, "./pics/frame0.jpg", idx)
