import numpy as np
from gunnar.core import Device, Array


SUPPORTED_ACTIVATIONS = ['sigmoid', 'relu', 'softmax']


class Dense:

    def __init__(self, inshape, outshape, activation='sigmoid'):
        assert activation in SUPPORTED_ACTIVATIONS
        self.activation = activation
        self.weight = None
        self.dweight = None
        self.weight_np = np.random.randn(outshape, inshape) * np.sqrt(2 / inshape)
        self.bias = None
        self.dbias = None
        self.bias_np = np.zeros((outshape, 1))
        self.y = None
        self.x = None
        self.gpu = None

    def __connect_device__(self, device: Device):
        self.gpu = device
        self.weight = self.gpu.array(self.weight_np)
        self.dweight = self.gpu.empty_array(self.weight_np.shape)
        self.bias = self.gpu.array(self.bias_np)
        self.dbias = self.gpu.empty_array(self.bias_np.shape)
        return True

    def __call__(self, x: Array):
        self.x = x
        y = (x @ self.weight.transpose()) % self.bias
        if self.activation == 'sigmoid':
            self.y = Array.sigmoid(y)
        if self.activation == 'relu':
            self.y = Array.relu(y)
        if self.activation == 'softmax':
            self.y = Array.softmax(y)
        return self.y

    def backward(self, err, lr):
        if self.activation == 'sigmoid':
            err = err * Array.dsigmoid(self.y)
        if self.activation == 'relu':
            err = err * Array.drelu(self.y)
        if self.activation == 'softmax':
            err = err * Array.dsoftmax(self.y)
        errT = err.transpose()
        ones = self.gpu.array(np.ones((self.x.shape[1], 1)))
        self.dbias = (ones @ errT).scale(lr)
        self.dweight = (self.x.transpose() @ errT).scale(lr)
        self.bias = self.bias + self.dbias
        self.weight = self.weight + self.dweight
        error = err @ self.weight
        return error
