import sys
sys.path.append('../')
import numpy as np
import time
from gunnar.core import Device, Array


if __name__ == '__main__':

    devices = Device.getDevices()
    dnames = [d for d in devices]

    print('Found devices:\n\t{}'.format('\n\t'.join(dnames)))
    print('\nUsing ', dnames[0])
    gpu = Device([devices[dnames[0]]], DTYPE=np.float32, TSK=32, TS=32, WPTM=4, WPTN=4)

    A = np.random.randn(4096, 4096).astype(np.float32)
    B = np.random.randn(4096, 4096).astype(np.float32)

    clA = gpu.array(A)
    clB = gpu.array(B)

    t1 = time.time()
    clC = clB @ clA.transpose()
    cl_time = time.time() - t1

    C = clC.to_cpu()

    t1 = time.time()
    R = np.matmul(A, B)
    cpu_time = time.time() - t1

    score = np.linalg.norm(R - C)

    print('error: ', score)
    print('GFLOPS:\n\tCPU: {}\n\tGPU: {}'.format(4096 ** 3 / cpu_time / 1e9,
                                                4096 ** 3 / cl_time / 1e9))
