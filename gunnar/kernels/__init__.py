import os

CL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'kernels')

with open(os.path.join(CL_DIR, 'mat.c'), 'r') as _f:
    MAT_KERNELS = _f.read()
