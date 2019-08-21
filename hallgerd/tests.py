import unittest
import logging
import numpy as np

from sklearn.datasets import make_classification

from hallgerd.core import Sequential
from hallgerd.layers import Dense
from gunnar.core import Device


class TestModels(unittest.TestCase):

    def test_mlp_classifier(self):
        from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
        from sklearn.metrics import classification_report, accuracy_score

        X, y = make_classification(n_samples=10000, n_features=200, n_informative=200, n_redundant=0, n_classes=5)
        y = OneHotEncoder(sparse=False, categories='auto').fit_transform(y.reshape((-1,1)))
        X = StandardScaler().fit_transform(X)
        y = y.T
        X = X.T
        devices = Device.getDevices()
        names = [k for k in devices]
        assert names
        cldevice = Device([devices[names[0]]])
        model = Sequential(device=cldevice, lr=1e-1, batch_size=1024, epochs=40, loss='cross_entropy', verbose=True)
        model.add(Dense(200, 200, activation='relu'))
        model.add(Dense(200, 5, activation='softmax'))
        model.fit(X, y) 
        yt = y.argmax(axis=0)
        ypp = model(X)
        yp = ypp.argmax(axis=0)
        score = accuracy_score(yt, yp)
        self.assertGreater(score, 0.8, msg='MLP wrong')


if __name__ == "__main__":
    unittest.main()
