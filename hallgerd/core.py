import logging
import numpy as np
from tqdm import tqdm

from hallgerd.losses import *
from gunnar.core import Device


SUPPORTED_LOSSES = ['mse', 'cross_entropy']


class Sequential:

    def __init__(self, device : Device, lr=1e-3, batch_size=4, epochs=1, loss='mse', verbose=False):
        assert loss in SUPPORTED_LOSSES
        self.gpu = device
        self.loss = loss
        self.lr = lr
        self.bs = batch_size
        self.epochs = epochs
        self.layers = list()
        self.history = {}
        self.verbose = verbose

    def add(self, layer):
        layer.__connect_device__(self.gpu)
        self.layers.append(layer)

    def __call__(self, x):
        xa = self.gpu.array(x.copy())
        for layer in self.layers:
            xa = layer(xa)
        return xa.to_cpu()

    def predict(self, X):
        yp = list()
        num_batches = np.ceil(X.shape[1] / self.bs)
        for x in np.array_split(X.T, num_batches):
            yp.append(self.__call__(x.T))
        return np.hstack(yp)

    def backward(self, y):
        y = y.copy()
        yp = self.layers[-1].y.to_cpu()
        if self.loss == 'mse':
            error = mse_delta(y, yp)
        if self.loss == 'cross_entropy':
            error = cross_entropy_delta(y, yp)
        error = self.gpu.array(error)
        for layer in reversed(self.layers):
            error = layer.backward(error, self.lr)
        return True

    def fit(self, X, y):
        assert X.shape[1] == y.shape[1]
        logging.info('fitting on {} data'.format(X.shape))
        num_batches = np.ceil(X.shape[1] / self.bs)
        self.history['loss'] = list()
        pbar = tqdm(range(self.epochs), disable=not self.verbose)
        for _ in pbar:
            for x, yt in zip(np.array_split(X.T, num_batches), np.array_split(y.T, num_batches)):
                _ = self.__call__(x.T)
                self.backward(yt.T)
            yp = self.predict(X)
            if self.loss == 'mse':
                loss = mse(y, yp)
            if self.loss == 'cross_entropy':
                loss = cross_entropy(y, yp)
            logging.info('train loss: {}'.format(loss))
            pbar.set_description('loss = {}'.format(loss))
            self.history['loss'].append(loss)

