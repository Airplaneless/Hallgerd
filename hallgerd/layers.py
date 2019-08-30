import numpy as np

from gunnar.core import Device, Array, Image, SUPPORTED_ACTIVATIONS


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
        self.weight_np = np.random.randn(out_channels*in_channels*kernel_size[0]*kernel_size[1], 1)
        self.weight_np *= np.sqrt(2 / self.weight_np.size)
        self.bias_np = np.zeros((in_channels*out_channels, 1))

    def __connect_device__(self, device: Device):
        self.gpu = device
        self.weight = self.gpu.image(self.weight_np, self.kernel_size, (self.out_channels, self.in_channels))
        self.dweight = self.gpu.empty_image(self.weight_np.shape, self.kernel_size, (self.out_channels, self.in_channels))
        self.bias = self.gpu.image(self.bias_np, self.kernel_size, (self.out_channels, self.in_channels))
        self.dbias = self.gpu.empty_image(self.bias_np.shape, self.kernel_size, (self.out_channels, self.in_channels))
        return True

    def __call__(self, x: Image):
        self.x = x
        y = x.conv2d(self.weight, padding=self.padding)
        if self.activation == 'linear':
            self.y = y.linear()
        if self.activation == 'sigmoid':
            self.y = y.sigmoid()
        if self.activation == 'relu':
            self.y = y.relu()
        if self.activation == 'softmax':
            self.y = y.softmax()
        return self.y

    def backward(self, err, lr):
        if self.activation == 'linear':
            err = self.y.dlinear() * err
        if self.activation == 'sigmoid':
            err = self.y.dsigmoid() * err
        if self.activation == 'relu':
            err = self.y.drelu() * err
        if self.activation == 'softmax':
            err = self.y.dsoftmax() * err
        fxs = self.x.image_shape[0] * 2 - 1
        fys = self.x.image_shape[1] * 2 - 1
        xpad = (fxs - self.weight.image_shape[0]) // 2
        ypad = (fys - self.weight.image_shape[1]) // 2
        xarea = (xpad, fxs - xpad)
        yarea = (ypad, fys - ypad)
        self.dweight = err.dconv2d(self.x, area=(xarea, yarea), padding=0).scale(lr)
        self.weight = self.weight + self.dweight
        error = err.conv2d(self.weight, padding=0, reverse=True)
        return error


class Dense(AbstractLayer):

    def __init__(self, inshape, outshape, activation='sigmoid'):
        super().__init__()
        assert activation in SUPPORTED_ACTIVATIONS
        self.in_shape = inshape
        self.out_shape = outshape
        self.activation = activation
        self.weight_np = np.random.randn(outshape, inshape)
        self.weight_np *= np.sqrt(2 / self.weight_np.size)
        self.bias_np = np.zeros((outshape, 1))

    def __call__(self, x: Array):
        self.x = x
        y = (x @ self.weight.transpose()) % self.bias
        if self.activation == 'linear':
            self.y = y.linear()
        if self.activation == 'sigmoid':
            self.y = y.sigmoid()
        if self.activation == 'relu':
            self.y = y.relu()
        if self.activation == 'softmax':
            self.y = y.softmax()
        return self.y

    def backward(self, err, lr):
        if self.activation == 'linear':
            err = err * self.y.dlinear()
        if self.activation == 'sigmoid':
            err = err * self.y.dsigmoid()
        if self.activation == 'relu':
            err = err * self.y.drelu()
        if self.activation == 'softmax':
            err = err * self.y.dsoftmax()
        errT = err.transpose()
        ones = self.gpu.array(np.ones((self.x.shape[1], 1)))
        self.dbias = (ones @ errT).scale(lr)
        self.dweight = (self.x.transpose() @ errT).scale(lr)
        self.bias = self.bias + self.dbias
        self.weight = self.weight + self.dweight
        error = err @ self.weight
        return error
