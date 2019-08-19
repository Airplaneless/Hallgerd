import sys
sys.path.append('../')
import numpy as np

import hallgerd
from hallgerd.core import Sequential
from hallgerd.layers import Dense
from gunnar.core import Device

from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


if __name__ == '__main__':

    d0 = np.fromfile('mnist-data/data0', dtype=np.uint8).reshape((1000, 28, 28))
    d1 = np.fromfile('mnist-data/data1', dtype=np.uint8).reshape((1000, 28, 28))
    d2 = np.fromfile('mnist-data/data2', dtype=np.uint8).reshape((1000, 28, 28))
    d3 = np.fromfile('mnist-data/data3', dtype=np.uint8).reshape((1000, 28, 28))
    d4 = np.fromfile('mnist-data/data4', dtype=np.uint8).reshape((1000, 28, 28))
    d5 = np.fromfile('mnist-data/data5', dtype=np.uint8).reshape((1000, 28, 28))
    d6 = np.fromfile('mnist-data/data6', dtype=np.uint8).reshape((1000, 28, 28))
    d7 = np.fromfile('mnist-data/data7', dtype=np.uint8).reshape((1000, 28, 28))
    d8 = np.fromfile('mnist-data/data8', dtype=np.uint8).reshape((1000, 28, 28))
    d9 = np.fromfile('mnist-data/data9', dtype=np.uint8).reshape((1000, 28, 28))
    dataX = np.concatenate((d0, d1, d2, d3, d4, d5, d6, d7, d8, d9))
    dataY = np.concatenate(([0]*1000, [1]*1000, [2]*1000, [3]*1000, [4]*1000, [5]*1000, [6]*1000, [7]*1000, [8]*1000, [9]*1000))

    indices = np.random.permutation(10000)
    dataX = dataX[indices]
    dataY = dataY[indices]
    dataX = dataX.reshape(10000, 784).T.astype(np.float32) / 255
    dataY = OneHotEncoder(categories='auto', sparse=False).fit_transform(dataY.reshape(-1, 1)).T

    devices = Device.getDevices()
    dnames = [d for d in devices]

    print('Found devices:\n\t{}'.format('\n\t'.join(dnames)))
    print('\nUsing ', dnames[0])
    gpu = Device([devices[dnames[0]]], DTYPE=np.float32, TS=32, TSK=32, WPTM=4, WPTN=4, TSM=128, TSN=128)

    model = Sequential(gpu, lr=1e-1, batch_size=1024, epochs=20, loss='cross_entropy', verbose=True)
    model.add(Dense(784, 512, activation='relu'))
    model.add(Dense(512, 512, activation='relu'))
    model.add(Dense(512, 10, activation='softmax'))
    model.fit(dataX, dataY)

    yt = dataY.argmax(axis=0)
    ypp = model.predict(dataX)
    yp = ypp.argmax(axis=0)
    print(classification_report(yt, yp))

#               precision    recall  f1-score   support

#            0       0.91      0.96      0.94      1000
#            1       0.93      0.95      0.94      1000
#            2       0.88      0.88      0.88      1000
#            3       0.91      0.79      0.84      1000
#            4       0.96      0.53      0.69      1000
#            5       0.90      0.77      0.83      1000
#            6       0.88      0.94      0.91      1000
#            7       0.95      0.79      0.86      1000
#            8       0.78      0.86      0.82      1000
#            9       0.57      0.94      0.71      1000

#     accuracy                           0.84     10000
#    macro avg       0.87      0.84      0.84     10000
# weighted avg       0.87      0.84      0.84     10000
