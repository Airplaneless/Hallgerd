import unittest
import logging
import numpy as np

from sklearn.datasets import make_classification
from gunnar.core import Device


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
        A = np.random.randn(420, 28).astype(np.float32)
        B = np.random.randn(420, 1).astype(np.float32)
        clA = gpu.array(A)
        clB = gpu.array(B)
        clC = clA % clB
        C = clC.to_cpu()
        score = np.linalg.norm((A + B) - C)
        self.assertLess(score, 0.1, msg='Wrong sumcol')


if __name__ == "__main__":
    unittest.main()
