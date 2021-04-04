import time
import numpy as np
from numpy.random import seed
seed(625742)
import cv2
import sys
import os
import signal
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import importlib
from keras.optimizers import rmsprop

modelname = sys.argv[1]
sys.path.append("./models")
print("Model: " + modelname + ".py")
deeplight = importlib.import_module(modelname)

def signal_handler(sig, frame):
    print('====================== You pressed Ctrl+C! =============================')
    should_save = raw_input("Save the model? ")
    if should_save == 'y' or should_save == 'Y':
        model.save_weights(weight_file)
        print("Saved the model")
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    #Confirm with bitwise comparison
    preds = (model.predict(x_test) >= 0.5).astype(np.uint8)
    print(preds.shape)
    print(y_test.shape)
    correct = 0.0
    total = 0.0
    for i in range(len(preds)):
        for k in range(len(preds[i])):
            total += 1
            if preds[i][k] == y_test[i][k]:
                correct += 1
    print("Confirm accuracy: " + str(correct/total))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.savefig(figure_file)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#-----------------------------------------------------------------
# $train_deeplight modelfile datafile iteration weightfile figurefile
#-----------------------------------------------------------------

data = np.load(sys.argv[2])
iteration = int(sys.argv[3])
resume = bool(sys.argv[4])
weight_file = "weights/" + modelname + ".h5"
figure_file = "graphs/" + modelname + ".pdf"
x_data = data['X']
y_label = data['Y']
print(x_data.shape)
print(y_label.shape)
indexes = np.arange(len(x_data))
# np.random.shuffle(indexes)
x_train = x_data[indexes[:int(0.99*len(x_data))],:,:,:]
y_train = y_label[indexes[:int(0.99*len(x_data))],:]
x_test = x_data[indexes[int(0.99*len(x_data)):],:,:,:]
y_test = y_label[indexes[int(0.99*len(x_data)):],:]
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_train -= 0.5
x_train *= 2.

x_test /= 255
x_test -= 0.5
x_test *= 2.

model = deeplight.deeplightmodel()
print(model.summary())
if resume and os.path.isfile(weight_file):
    model.load_weights(weight_file)
    print("Load a previously saved model at: " + weight_file)

opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='binary_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])
history = model.fit(x_train, y_train,
          validation_split=0.2,
          batch_size=16,
          epochs=iteration,
          shuffle=False)
model.save_weights(weight_file)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#Confirm with bitwise comparison
preds = (model.predict(x_test) >= 0.5).astype(np.uint8)
print(preds.shape)
print(y_test.shape)
correct = 0.0
total = 0.0
for i in range(len(preds)):
    for k in range(len(preds[i])):
        total += 1
        if preds[i][k] == y_test[i][k]:
            correct += 1
print("Confirm accuracy: " + str(correct/total))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.savefig(figure_file)

