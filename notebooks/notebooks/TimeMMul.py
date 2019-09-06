import sys
sys.path.append('../repos/Hallgerd/')
import numpy as np
import torch
import time
from gunnar.core import Device


if __name__ == '__main__':

    devices = Device.getDevices()
    dnames = [d for d in devices]

    print('Found devices:\n\t{}'.format('\n\t'.join(dnames)))
    print('\nUsing ', dnames[0])
    gpu = Device([devices[dnames[0]]], DTYPE=np.float32, TSK=32, TS=32, WPTM=4, WPTN=4, TPM=128, TPN=128)

    A = np.random.randn(4096, 4096).astype(np.float32)
    B = np.random.randn(4096, 4096).astype(np.float32)

    clA = gpu.array(A)
    clB = gpu.array(B)
    
    tA = torch.tensor(A).cuda()
    tB = torch.tensor(B).cuda()

    t1 = time.time()
    clC = clB @ clA.transpose()
    cl_time = time.time() - t1
    
    t1 = time.time()
    tC = tA @ tB
    t_time = time.time() - t1

    C = clC.to_cpu()

    t1 = time.time()
    R = A @ B
    cpu_time = time.time() - t1

    score = np.linalg.norm(R - C)

    print('error: ', score)
    print('Time, s:\n\tCPU: {}\n\tGunnar: {}\n\tPyTorch: {}'.format(cpu_time, cl_time, t_time))
