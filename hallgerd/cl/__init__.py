import os

CL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'cl')

with open(os.path.join(CL_DIR, 'gd.cl'), 'r') as _f:
    GD_CL_KERNELS = _f.read()

with open(os.path.join(CL_DIR, 'mat_kernels.cl'), 'r') as _f:
    MAT_CL_KERNELS = _f.read()
