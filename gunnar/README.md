# Gunnar
### Array handler for OpenCL devices

## Usage

Get dict of available OpenCL devices:
```python
In [1]: from gunnar.core import Device
        devices = Device.getDevices()
In [2]: devices
Out[2]: {'GeForce GTX 660': <pyopencl.Device 'GeForce GTX 660' on 'NVIDIA CUDA' at 0x2694714a8e0>}
```

Creation of OpenCL device with specified array type
```python
In [3]: gpu = Device(devices['GeForce GTX 660'], DTYPE=np.float32, WPTM=4, WPTN=4)
```

`gunnar.core.Device` has also attributes `TS`, `TSM`, `TSN`, `WPTM`, `WPTN` (see https://cnugteren.github.io/tutorial/pages/page8.html)

This attributes must satisfy the conditions: 

`TS` < CL_DEVICE_MAX_WORK_GROUP_SIZE

`TSM` / `WPTM` < CL_DEVICE_MAX_WORK_GROUP_SIZE

`TSN` / `WPTN` < CL_DEVICE_MAX_WORK_GROUP_SIZE

Creation of OpenCL arrays from numpy arrays
```python
In [5]: A = np.random.randn(420, 690)
        B = np.random.randn(690, 228)
In [6]: clA = gpu.array(A)
        clB = gpu.array(B)
```
Matrix operation and copy to numpy array
```python
In [7]: clC = clB @ clA.transpose() # equivalent to A @ B in numpy
In [8]: C = clC.to_cpu() # returns numpy.ndarray
```
## Performance 

![mmulFLOPS](https://drive.google.com/uc?id=19BSTtkUd1Kc_oON4e4dwl43l4xDYQJqL)
Performance of matrix multiplication with NumPy, PyTorch and Gunnar with copy result back to host (matrix sizes: 4096x4096x4096, dtype=float32)
