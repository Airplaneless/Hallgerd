import numpy as np

SUPPORTED_ACTIVATIONS = ['sigmoid', 'relu', 'softmax']


def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)


def d_softmax(sx):
    return sx * (1 - sx)


def relu(x):
    x[x <= 0] = 0
    return x


def d_relu(x):
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(sx):
    return sx * (1 - sx)


class Dense:

    def __init__(self, inshape, outshape, activation='sigmoid'):
        self.activation = activation
        assert activation in SUPPORTED_ACTIVATIONS
        self.weight = np.random.randn(outshape, inshape) * np.sqrt(2 / inshape)
        self.bias = np.zeros((outshape, 1))
        self.y = None
        self.x = None

    def __call__(self, x):
        y = np.matmul(self.weight, x) + self.bias
        if self.activation == 'sigmoid':
            self.y = sigmoid(y)
        if self.activation == 'relu':
            self.y = relu(y)
        if self.activation == 'softmax':
            self.y = softmax(y)
        self.x = x
        return self.y

    def backprop(self, err, lr):
        if self.activation == 'sigmoid':
            err = err * d_sigmoid(self.y)
        if self.activation == 'relu':
            err = err * d_relu(self.y)
        if self.activation == 'softmax':
            err = err * d_softmax(self.y)
        self.bias += np.matmul(err, np.ones((self.x.shape[1], 1))) * lr
        self.weight += np.matmul(err, self.x.T) * lr
        error = np.matmul(self.weight.T, err)
        return error