# Gunnar
### Array handler for OpenCL devices

## Usage

Get dict of available OpenCL devices:

    In [1]: from gunnar.core import Device
            devices = Device.getDevices()
    In [2]: devices
    Out[2]: {'GeForce GTX 660': <pyopencl.Device 'GeForce GTX 660' on 'NVIDIA CUDA' at 0x2694714a8e0>}

Creation of OpenCL device with specified array type

    In [3]: gpu = Device(devices['GeForce GTX 660'], DTYPE=np.float32)

Creation of OpenCL arrays from numpy arrays

    In [5]: A = np.random.randn(420, 690)
            B = np.random.randn(690, 228)
    In [6]: clA = gpu.array(A)
            clB = gpu.array(B)

Matrix operation and copy to numpy array

    In [7]: clC = clB @ clA.transpose()
    In [8]: C = clC.to_cpu() # returns numpy.ndarray

## Performance 

![mmulFLOPS](https://drive.google.com/uc?id=19BSTtkUd1Kc_oON4e4dwl43l4xDYQJqL)
Performance of matrix multiplication with NumPy, PyTorch and Gunnar with copy result back to host (matrix sizes: 4096x4096x4096, dtype=float32)
