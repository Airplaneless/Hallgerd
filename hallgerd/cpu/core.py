import numpy as np
from tqdm import tqdm

SUPPORTED_LOSSES = ['mse', 'cross_entropy']


def mse(yt, yp):
    return np.linalg.norm(yt - yp)


def mse_delta(yt, yp):
    return 2 * (yt - yp)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def cross_entropy(yt, yp):
    # p = softmax(yp)
    loss = -np.sum(yt * yp)
    return loss


def cross_entropy_delta(yt, yp):
    p = softmax(yp)
    return yt - yp


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
                loss = mse(y, self.__call__(X))
            if self.loss == 'cross_entropy':
                loss = cross_entropy(y, self.__call__(X))
            self.history['loss'].append(loss)
