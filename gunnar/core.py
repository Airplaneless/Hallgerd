import numpy as np
import pyopencl as cl
from gunnar.kernels import MAT_KERNELS


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Array:

    def __init__(self, ctx, queue, prg):
        self.ctx = ctx
        self.queue = queue
        self.prg = prg
        self.buffer = None


class Device(metaclass=Singleton):

    def __init__(self, devices: list):
        self.ctx = cl.Context(devices)
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, MAT_KERNELS).build()

    def array(self, cpu_arr: np.ndarray):


    @staticmethod
    def getDevices():
        res = dict()
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError('No OpenCL platforms')
        else:
            for p in platforms:
                devices = p.get_devices()
                if devices:
                    for d in devices:
                        res[d.name] = d
        return res


