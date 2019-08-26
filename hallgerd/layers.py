import numpy as np

from gunnar.core import Device, Array

SUPPORTED_ACTIVATIONS = ['sigmoid', 'relu', 'softmax']


class AbstractLayer:

    def __init__(self):
        self.weight = None
        self.dweight = None
        self.weight_np = None
        self.bias = None
        self.dbias = None
        self.bias_np = None
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


class Conv2D(AbstractLayer):

    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=0, activation='sigmoid'):
        super().__init__()
        assert activation in SUPPORTED_ACTIVATIONS
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight_np = np.random.randn(out_channels*in_channels*kernel_size[0]*kernel_size[1], 1) * np.sqrt(2 / kernel_size[0]*kernel_size[1])
        self.bias_np = np.zeros((in_channels*out_channels, 1))

    def __connect_device__(self, device: Device):
        self.gpu = device
        self.weight = self.gpu.image(self.weight_np, self.kernel_size, (self.out_channels, self.in_channels))
        self.dweight = self.gpu.empty_image(self.weight_np.shape, self.kernel_size, (self.out_channels, self.in_channels))
        self.bias = self.gpu.image(self.bias_np, self.kernel_size, (self.out_channels, self.in_channels))
        self.dbias = self.gpu.empty_image(self.bias_np.shape, self.kernel_size, (self.out_channels, self.in_channels))
        return True

    def __call__(self, x: Array):
        self.x = x
        y = x.conv2d(self.weight, padding=self.padding)
        if self.activation == 'sigmoid':
            self.y = y.sigmoid()
        if self.activation == 'relu':
            self.y = y.relu()
        if self.activation == 'softmax':
            self.y = y.softmax()
        return self.y

    def backward(self, err, lr):
        if self.activation == 'sigmoid':
            err = self.y.dsigmoid() * err
        if self.activation == 'relu':
            err = self.y.drelu() * err
        if self.activation == 'softmax':
            err = self.y.dsoftmax() * err
        self.dweight = err.fconv2d(self.x, padding=self.padding)
        xpad = (self.dweight.image_shape[0] - self.weight.image_shape[0]) // 2
        ypad = (self.dweight.image_shape[1] - self.weight.image_shape[1]) // 2
        xarea = (xpad, self.dweight.image_shape[0] - xpad)
        yarea = (ypad, self.dweight.image_shape[1] - ypad)
        self.dweight = self.dweight.crop((xarea, yarea)).scale(lr / (self.kernel_size[0]*self.kernel_size[1]))
        self.weight = self.weight + self.dweight
        error = err.conv2d(self.weight, padding=self.padding, reverse=True)
        return error


class Dense(AbstractLayer):

    def __init__(self, inshape, outshape, activation='sigmoid'):
        super().__init__()
        assert activation in SUPPORTED_ACTIVATIONS
        self.in_shape = inshape
        self.out_shape = outshape
        self.activation = activation
        self.weight_np = np.random.randn(outshape, inshape) * np.sqrt(2 / inshape)
        self.bias_np = np.zeros((outshape, 1))

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
        self.dbias = (ones @ errT).scale(lr / self.in_shape)
        self.dweight = (self.x.transpose() @ errT).scale(lr / self.in_shape)
        self.bias = self.bias + self.dbias
        self.weight = self.weight + self.dweight
        error = err @ self.weight
        return error
