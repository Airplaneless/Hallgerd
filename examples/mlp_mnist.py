import sys
sys.path.append('../')
import numpy as np

import hallgerd
from hallgerd.core import Sequential
from hallgerd.layers import Dense
from gunnar.core import Device

import keras
from keras.datasets import mnist
from sklearn.metrics import classification_report


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float64')
    x_test = x_test.astype('float64')
    x_train /= 255
    x_test /= 255

    x_test = x_test.T
    x_train = x_train.T
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_train = y_train.T
    y_test = y_test.T

    devices = Device.getDevices()
    dnames = [d for d in devices]

    print('Found devices:\n\t{}'.format('\n\t'.join(dnames)))
    print('\nUsing ', dnames[0])
    gpu = Device([devices[dnames[0]]], DTYPE=np.float32, TS=32, TSK=32, WPTM=4, WPTN=4, TSM=128, TSN=128)

    model = Sequential(gpu, lr=1e-1, batch_size=1024, epochs=5, loss='cross_entropy', verbose=True)
    model.add(Dense(784, 512, activation='relu'))
    model.add(Dense(512, 512, activation='relu'))
    model.add(Dense(512, 10, activation='softmax'))
    model.fit(x_train, y_train)

    yt = y_test.argmax(axis=0)
    ypp = model.predict(x_test)
    yp = ypp.argmax(axis=0)
    print(classification_report(yt, yp))

