# ![logo](https://drive.google.com/uc?id=1Pe_qvgtWcw3l_vTzwsIjdNVtovzlQAeZ)
### Deep learning framework for OpenCL

Draft of dl framework for OpenCL.
There is only Dense layer for now. 
Supports sigmoid, relu, softmax activations, MSE and categorical cross entropy loss functions

## Usage

Select OpenCL device and init model parameters (see gunnar README)

    In [1]: from hallgerd.core import Sequential
            model = Sequential(device=devices['GeForce GTX 660'], lr=1e-1, batch_size=1024, epochs=40, loss='cross_entropy', verbose=True)

Add layers

    In [2]: model.add(Dense(200, 200, activation='relu'))
            model.add(Dense(200, 5, activation='softmax'))

Train model

    In [3]: model.fit(X, y) # here X.shape = (feature size, dataset size)
                            # y.shape = (output size, dataset size)

Prediction

    In [4]: yp = model(X)


## Performance

![mmulFLOPS](https://drive.google.com/uc?id=1NkNHZIpDmFg7BvZzzxeeS1Y4UOiuOUPD)
Training MLP with PyTorch and Hallgerd
