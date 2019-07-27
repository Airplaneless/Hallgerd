import numpy as np
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(sx):
    return sx * (1 - sx)


class MLP:

    def __init__(self, lr=1e-3, batch_size=4, epochs=1):
        self.lr = lr
        self.bs = batch_size
        self.epochs = epochs
        self.layers = list()

    def add(self, inshape, outshape):
        self.layers.append(LayerMLP(inshape, outshape))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backprop(self, y):
        err = 2 * (y - self.layers[-1].y)
        for layer in reversed(self.layers):
            err = layer.backprop(err, self.lr)
        return True

    def fit(self, X, y):
        assert X.shape[1] == y.shape[1]
        num_batches = np.ceil(X.shape[1] / self.bs)
        for _ in tqdm(range(self.epochs)):
            for x, yt in zip(np.array_split(X.T, num_batches), np.array_split(y.T, num_batches)):
                _ = self.__call__(x.T)
                self.backprop(yt.T)


class LayerMLP:

    def __init__(self, inshape, outshape):
        self.weight = np.random.randn(outshape, inshape)
        self.bias = np.zeros((outshape, 1))
        self.y = None
        self.x = None

    def __call__(self, x):
        y = np.matmul(self.weight, x) + self.bias
        self.y = sigmoid(y)
        self.x = x
        return self.y

    def backprop(self, err, lr):
        err = err * sigmoid_der(self.y)
        self.bias += np.matmul(err, np.ones((self.x.shape[1], 1))) * lr
        self.weight += np.matmul(err, self.x.T) * lr
        error = np.matmul(self.weight.T, err)
        return error


if __name__ == "__main__":

    def xor(x1, x2):
        if 1 / (x1 * x1) + 1 / (x2 * x2) > 4:
            return 1
        return 0


    vxor = np.vectorize(xor)
    X = np.random.randn(10000, 2, )
    y = vxor(X[:, 0], X[:, 1])
    X = X.T
    y = y.reshape((1, -1))
    mlp = MLP(lr=0.1, batch_size=1, epochs=4)
    mlp.add(2, 4)
    mlp.add(4, 1)
    mlp.fit(X, y)

