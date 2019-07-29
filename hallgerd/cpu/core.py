import numpy as np
from tqdm import tqdm

SUPPORTED_LOSSES = ['mse', 'cross_entropy']


def mse(yt, yp):
    return np.linalg.norm(yt - yp)


def mse_delta(yt, yp):
    return 2 * (yt - yp)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)


def _softmax(X):
    expvx = np.exp(X - np.max(X, axis=1)[..., np.newaxis])
    return expvx/np.sum(expvx, axis=1, keepdims=True)


def cross_entropy(yt, yp):
    m = yt.shape[1]
    dyt = yt.argmax(axis=0)
    p = softmax(yp)
    log_likelihood = - np.log(p[dyt, range(m)])
    loss = np.sum(log_likelihood) / m
    return loss


def cross_entropy_delta(yt, yp):
    m = yt.shape[1]
    dyt = yt.argmax(axis=0)
    grad = softmax(yp)
    grad[dyt, range(m)] -= 1
    # grad = grad / m
    return -grad


class Sequential:

    def __init__(self, lr=1e-3, batch_size=4, epochs=1, loss='mse'):
        assert loss in SUPPORTED_LOSSES
        self.loss = loss
        self.lr = lr
        self.bs = batch_size
        self.epochs = epochs
        self.loss = loss
        self.layers = list()
        self.history = {}

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backprop(self, y):
        yp = self.layers[-1].y
        if self.loss == 'mse':
            error = mse_delta(y, yp)
        if self.loss == 'cross_entropy':
            error = cross_entropy_delta(y, yp)
        for layer in reversed(self.layers):
            error = layer.backprop(error, self.lr)
        return True

    def fit(self, X, y):
        assert X.shape[1] == y.shape[1]
        num_batches = np.ceil(X.shape[1] / self.bs)
        self.history['loss'] = list()
        for _ in tqdm(range(self.epochs)):
            for x, yt in zip(np.array_split(X.T, num_batches), np.array_split(y.T, num_batches)):
                _ = self.__call__(x.T)
                self.backprop(yt.T)
            if self.loss == 'mse':
                loss = mse(y, self.__call__(X)) / y.shape[1]
            if self.loss == 'cross_entropy':
                loss = cross_entropy(y, self.__call__(X)) / y.shape[1]
            self.history['loss'].append(loss)