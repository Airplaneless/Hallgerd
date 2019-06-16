import os

HALLGERD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'hallgerd')
CL_DIR = os.path.join(HALLGERD_DIR, 'cl')
GD_CL_KERNELS = os.path.join(CL_DIR, 'gd.cl')