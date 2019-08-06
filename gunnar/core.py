import numpy as np
import pyopencl as cl

from gunnar.kernels import MAT_KERNELS

DTYPE = np.float32
TSM = 128
TSN = TSM
TSK = 16
WPTM = 8
WPTN = 8
BST = 512


def guardShapes(M):
    if (M.shape[1] % TSN == 0) and (M.shape[0] % TSM == 0):
        return M.copy()
    else:
        nMx, nMy = M.shape[0], M.shape[1]
        if nMx % TSM != 0:
            nMx = (nMx // TSM + 1) * TSM
        if nMy % TSN != 0:
            nMy = (nMy // TSN + 1) * TSN
        nM = np.zeros((nMx, nMy), dtype=DTYPE)
        nM[:M.shape[0], :M.shape[1]] = M.copy()
        return nM.copy()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Array:

    def __init__(self, ctx, queue, prg, buffer, shape, bshape):
        self.ctx = ctx
        self.queue = queue
        self.prg = prg
        self.buffer = buffer
        assert len(shape) == 2
        assert len(bshape) == 2
        self.shape = shape
        self.bshape = bshape
    
    def to_cpu(self):
        x = np.empty(self.bshape, dtype=np.float32)
        event = cl.enqueue_copy(self.queue, x, self.buffer)
        cl.wait_for_events([event,])
        return x[:self.shape[0], :self.shape[1]]

    def __mul__(self, other):
        assert self.shape[0] == other.shape[0]
        assert self.shape[1] == other.shape[1]
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        out_shape = (self.shape[0], self.shape[1])
        cpu_earr = np.empty((M, N), dtype=DTYPE)
        resbuff = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.ctx, self.queue, self.prg, resbuff, out_shape, (M, N))
        global_sizes = (int(M), int(N))
        local_sizes = (int(TSK), int(TSK))
        event = self.prg.matdot(self.queue, global_sizes, local_sizes, M, N, self.buffer, other.buffer, res.buffer)
        cl.wait_for_events([event,])
        return res

    def scale(self, scalar: float):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        out_shape = (self.shape[0], self.shape[1])
        cpu_earr = np.empty((M, N), dtype=DTYPE)
        resbuff = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.ctx, self.queue, self.prg, resbuff, out_shape, (M, N))
        scalar = DTYPE(scalar)
        global_sizes = (int(M), int(N))
        local_sizes = (int(TSK), int(TSK))
        event = self.prg.matscale(self.queue, global_sizes, local_sizes, M, N, self.buffer, scalar, res.buffer)
        cl.wait_for_events([event,])
        return res

    def __pow__(self, other):
        assert self.shape[0] == other.shape[0]
        M = np.int32(self.bshape[1])
        K = np.int32(self.bshape[0])
        N = np.int32(other.bshape[1])
        out_shape = (other.shape[1], self.shape[1])
        cpu_earr = np.empty((N, M), dtype=DTYPE)
        resbuff = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.ctx, self.queue, self.prg, resbuff, out_shape, (N, M))
        global_sizes = (int(M/WPTM), int(N/WPTN))
        local_sizes = (int(TSM/WPTM), int(TSN/WPTN))
        event = self.prg.matmul(self.queue, global_sizes, local_sizes, M, N, K, self.buffer, other.buffer, res.buffer)
        cl.wait_for_events([event,])
        return res

    def transpose(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        out_shape = (self.shape[1], self.shape[0])
        cpu_earr = np.empty((N, M), dtype=DTYPE)
        resbuff = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.ctx, self.queue, self.prg, resbuff, out_shape, (N, M))
        global_sizes = (int(N), int(M))
        local_sizes = (int(TSK), int(TSK))
        event = self.prg.transpose(self.queue, global_sizes, local_sizes, N, M, self.buffer, res.buffer)
        cl.wait_for_events([event,])
        return res

    def __add__(self, other):
        assert self.shape[0] == other.shape[0]
        assert self.shape[1] == other.shape[1]
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        out_shape = (self.shape[0], self.shape[1])
        cpu_earr = np.empty((M, N), dtype=DTYPE)
        resbuff = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.ctx, self.queue, self.prg, resbuff, out_shape, (M, N))
        global_sizes = (int(M), int(N))
        local_sizes = (int(TSK), int(TSK))
        event = self.prg.matsum(self.queue, global_sizes, local_sizes, M, N, self.buffer, other.buffer, res.buffer)
        cl.wait_for_events([event,])
        return res

    def __lt__(self, other):
        # assert self.shape[0] == other.shape[0]
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        K = np.int32(other.bshape[1])
        out_shape = (self.shape[0], self.shape[1])
        cpu_earr = np.empty((M, N), dtype=DTYPE)
        resbuff = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.ctx, self.queue, self.prg, resbuff, out_shape, (M, N))
        global_sizes = (int(M), int(N))
        local_sizes = (int(TSK), int(TSK))
        event = self.prg.matpluscol(self.queue, global_sizes, local_sizes, M, N, K, self.buffer, other.buffer, res.buffer)
        cl.wait_for_events([event,])
        return res


class Device(metaclass=Singleton):

    def __init__(self, devices: list):
        self.ctx = cl.Context(devices)
        self.queue = cl.CommandQueue(self.ctx)
        options = "-DTSM={} -DTSN={} -DTSK={} -DWPTM={} -DWPTN={}".format(TSM, TSN, TSK, WPTM, WPTN)
        self.prg = cl.Program(self.ctx, MAT_KERNELS).build(options)

    def array(self, cpu_arr: np.ndarray):
        arr = guardShapes(cpu_arr)
        buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
        return Array(self.ctx, self.queue, self.prg, buffer, cpu_arr.shape, arr.shape)

    def empty_array(self, shape):
        cpu_earr = np.empty(shape, dtype=DTYPE)
        earr = guardShapes(cpu_earr)
        buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=earr.nbytes)
        return Array(self.ctx, self.queue, self.prg, buffer, cpu_earr.shape, earr.shape)

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


