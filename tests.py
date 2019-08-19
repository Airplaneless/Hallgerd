import unittest
import logging
import numpy as np

from sklearn.datasets import make_classification

from hallgerd.core import Sequential
from hallgerd.layers import Dense
from gunnar.core import Device, Array


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)


def d_softmax(sx):
    return sx * (1 - sx)


def relu(x):
    x[x <= 0] = 0
    return x


def d_relu(x):
    x = relu(x)
    x[x > 0] = 1.0
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(sx):
    return sx * (1 - sx)


class TestGunnar(unittest.TestCase):

    def test_sigmoid(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(69, 690)
        clA = gpu.array(A)
        clC = clA.sigmoid()
        C = clC.to_cpu()
        score = np.linalg.norm(sigmoid(A) - C)
        self.assertLess(score, 0.1, msg='Wrong sigmoid')

    def test_dsigmoid(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(69, 690)
        clA = gpu.array(A)
        clC = clA.dsigmoid()
        C = clC.to_cpu()
        score = np.linalg.norm(d_sigmoid(A) - C)
        self.assertLess(score, 0.1, msg='Wrong dsigmoid')

    def test_relu(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(69, 690)
        clA = gpu.array(A)
        clC = clA.relu()
        C = clC.to_cpu()
        score = np.linalg.norm(relu(A) - C)
        self.assertLess(score, 0.1, msg='Wrong relu')

    def test_drelu(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(69, 690)
        clA = gpu.array(A)
        clC = clA.drelu()
        C = clC.to_cpu()
        score = np.linalg.norm(d_relu(A) - C)
        self.assertLess(score, 0.1, msg='Wrong drelu')

    def test_softmax(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(690, 69)
        clA = gpu.array(A)
        clC = clA.softmax()
        C = clC.to_cpu()
        score = np.linalg.norm(softmax(A) - C)
        self.assertLess(score, 0.1, msg='Wrong softmax')

    def test_dsoftmax(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(69, 690)
        clA = gpu.array(A)
        clC = clA.dsoftmax()
        C = clC.to_cpu()
        score = np.linalg.norm(d_softmax(A) - C)
        self.assertLess(score, 0.1, msg='Wrong dsoftmax')

    def test_sum(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(69, 690).astype(np.float32)
        B = np.random.randn(69, 690).astype(np.float32)
        clA = gpu.array(A)
        clB = gpu.array(B)
        clC = clA + clB
        C = clC.to_cpu()
        score = np.linalg.norm((A+B) - C)
        self.assertLess(score, 0.1, msg='Wrong mat sum')

    def test_transpose(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(700, 100).astype(np.float32)
        clA = gpu.array(A)
        clB = clA.transpose()
        B = clB.to_cpu()
        score = np.linalg.norm(A.T - B)
        self.assertLess(score, 0.1, msg='Wrong transpose')

    def test_mmul(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(420, 690).astype(np.float32)
        B = np.random.randn(690, 228).astype(np.float32)
        clA = gpu.array(A)
        clB = gpu.array(B)
        clC = clB @ clA.transpose()
        C = clC.to_cpu()
        score = np.linalg.norm(np.matmul(A, B) - C)
        self.assertLess(score, 0.1, msg='Wrong matmul')

    def test_dot(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(690, 109)
        B = np.random.randn(690, 109)
        clA = gpu.array(A)
        clB = gpu.array(B)
        clC = clA * clB
        C = clC.to_cpu()
        score = np.linalg.norm((A*B) - C)
        self.assertLess(score, 0.1, msg='Wrong mat sum')

    def test_scale(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        A = np.random.randn(628, 128)
        s = 42
        clA = gpu.array(A)
        clB = clA.scale(s)
        B = clB.to_cpu()
        score = np.linalg.norm(A * s - B)
        self.assertLess(score, 0.1, msg='Wrong scale')

    def test_sumcol(self):
        devices = Device.getDevices()
        device = list(devices.keys())[0]
        gpu = Device([devices[device]])
        # A = np.zeros((10, 128)).astype(np.float32)
        # B = np.arange(10).reshape((-1,1)).astype(np.float32)
        A = np.random.randn(420, 28).astype(np.float32)
        B = np.random.randn(420, 1).astype(np.float32)
        clA = gpu.array(A)
        clB = gpu.array(B)
        clC = clA % clB
        C = clC.to_cpu()
        score = np.linalg.norm((A + B) - C)
        self.assertLess(score, 0.1, msg='Wrong sumcol')


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
        cldevice = Device([devices[names[0]]], TSK=32, WPTM=4, WPTN=4)
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
