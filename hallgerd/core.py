import logging
import numpy as np
import pyopencl as cl
from tqdm import tqdm

from hallgerd.cl import MAT_CL_KERNELS
from hallgerd.layers import Dense


class Sequential:

    def __init__(self, lr=1e-3, batch_size=1, epochs=1):
        self.lr = lr
        self.bs = batch_size
        # self._batches = None
        self.epochs = epochs
        self.layers = list()
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, MAT_CL_KERNELS).build()

    def add(self, layer):
        layer.__connect_context__(self.ctx, self.queue, self.prg)
        self.layers.append(layer)

    def __call__(self, x):
        _batches = x.shape[1]
        x = x.copy().astype(np.float64)
        x_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        for layer in self.layers:
            x_cl = layer(x_cl, batches=_batches)
        out_np = np.empty((self.layers[-1].out_shape, _batches), dtype=np.float64)
        cl.enqueue_copy(self.queue, out_np, x_cl)
        return out_np

    def backprop(self, y):
        y = y.copy().astype(np.float64)
        yp = np.empty((self.layers[-1].out_shape, y.shape[1]), dtype=np.float64)
        cl.enqueue_copy(self.queue, yp, self.layers[-1].output_cl)
        error = 2 * (y - yp)
        error_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=error)
        for layer in reversed(self.layers):
            error_cl = layer.backprop(error_cl, self.lr)
        return True

    def fit(self, X, y):
        assert X.shape[1] == y.shape[1]
        num_batches = np.ceil(X.shape[1] / self.bs)
        for _ in tqdm(range(self.epochs)):
            for x, yt in zip(np.array_split(X.T, num_batches), np.array_split(y.T, num_batches)):
                _ = self.__call__(x.T)
                self.backprop(yt.T)


if __name__ == '__main__':

    def xor(x1, x2):
        if 1 / (x1 * x1) + 1 / (x2 * x2) > 4:
            return 1
        return 0


    vxor = np.vectorize(xor)
    X = np.random.randn(10000, 2, )
    y = vxor(X[:, 0], X[:, 1])
    X = X.T
    y = y.reshape((1, -1))

    model = Sequential(batch_size=256, epochs=5)
    model.add(Dense(2, 4, activation='sigmoid'))
    model.add(Dense(4, 1, activation='softmax'))
    # model.fit(X, y)

    # X = np.random.random((2, 30)).astype(np.float64)
    # X = X.copy()
    print(model(X[:, 0:5]))
    # print(model(X[:, 0:20]))
    # print(model(x))
