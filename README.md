# ![logo](https://drive.google.com/uc?id=1Pe_qvgtWcw3l_vTzwsIjdNVtovzlQAeZ)
## Deep learning framework for OpenCL

Draft of DL framework for OpenCL.
There is only Dense layer. 
Supports sigmoid, relu, softmax activations, MSE and categorical cross entropy loss functions.

# Prerequisites
`Hallgerd` and `Gunnar` modules uses `pyopencl` to operate with OpenCL kernels. Before `pyopencl` installation you must have compatible OpenCL drivers and OpenCL library and headers on your system.

## Windows
You can check if drivers with OpenCL support installed by `GPU Caps Viewer`
### Intel
Intel Graphics drivers include OpenCL supports. However, if Windows 10 automaticaly installed Intel HD drives, you may reinstall them manually from official Intel site.

OpenCL library and headers included in Intel SDK for OpenCL applications: 
https://software.intel.com/en-us/opencl-sdk/choose-download

After unzip downloaded arcive you can just install      

    .\installs\opencl\intel_sdk_for_opencl_2019_x64_setup\intel_sdk_for_opencl_2019_x64_setup.msi

After installation you need add pathes to include and lib directories in system variables.

Create system variables 

INCLUDE=`C:\Program Files (x86)\Intel\OpenCL SDK\7.0\include`

and LIB=`C:\Program Files (x86)\Intel\OpenCL SDK\7.0\include`

### Nvidia
Nvidia drivers include OpenCL supports.

OpenCL library and headers included in CUDA Toolkit.

After installation CUDA Toolkit, add include and lib directories to system variables:

INCLUDE=`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include`

LIB=`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64`

## Linux

    _

# Installation

    $ git clone https://github.com/Airplaneless/Hallgerd.git  --branch 0.1.0
    $ cd Hallgerd
    $ python setup install

To make sure that all works

    $ python -m hallgerd.tests
    $ python -m gunnar.tests

# Usage

Select OpenCL device and init model parameters (see gunnar README)

```python
In [1]: from hallgerd.core import Sequential
        gpu = Device(devices['GeForce GTX 660'], DTYPE=np.float32)
        model = Sequential(device=gpu, lr=1e-1, batch_size=1024, epochs=40, loss='cross_entropy')
```

Add layers
```python
In [2]: model.add(Dense(200, 200, activation='relu'))
        model.add(Dense(200, 5, activation='softmax'))
```
Train model
```python
In [3]: model.fit(X, y) # here X.shape = (feature size, dataset size)
                        # y.shape = (output size, dataset size)
```
Prediction
```python
In [4]: yp = model(X)
```

# Performance

![mmulFLOPS](https://drive.google.com/uc?id=1NkNHZIpDmFg7BvZzzxeeS1Y4UOiuOUPD)
Training MLP with PyTorch and Hallgerd
