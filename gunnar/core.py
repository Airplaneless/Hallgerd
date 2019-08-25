import numpy as np
import pyopencl as cl

from functools import reduce

from gunnar.kernels import MAT_KERNELS
from hallgerd.losses import softmax

SUPPORTED_ACTIVATIONS = ['sigmoid', 'relu', 'softmax']
DTYPES = {np.float16 : 'half', np.float32 : 'float', np.float64 : 'double'}


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Array:

    def __init__(self, device, buffer, shape, bshape):
        self.device = device
        self.buffer = buffer
        assert len(bshape) == 2
        self.shape = shape
        self.bshape = bshape

    def sigmoid(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.sigmoid(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def dsigmoid(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.dsigmoid(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def relu(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.relu(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def drelu(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.drelu(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def softmax(self):
        # TODO: softmax on kernels
        nx = self.to_cpu()
        nres = softmax(nx).astype(self.device.DTYPE)
        out_shape = nres.shape
        nres = self.device.guardShapes(nres)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=nres)
        res = Array(self.device, resbuff, out_shape, self.bshape)
        return res

    def dsoftmax(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.dsoftmax(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self
    
    def to_cpu(self):
        x = np.empty(self.bshape, dtype=self.device.DTYPE)
        event = cl.enqueue_copy(self.device.queue, x, self.buffer)
        cl.wait_for_events([event, ])
        return x[:self.shape[0], :self.shape[1]]

    def conv2d(self, filter, fshape, imgshape, channels, padding=0):
        imgX, imgY = imgshape
        imgIC, imgOC = channels
        assert len(fshape) == 2
        assert self.shape[0] == imgX * imgY * imgIC
        assert filter.shape[0] == fshape[0] * fshape[1] * imgIC * imgOC
        assert filter.shape[1] == 1

        cI = np.int32(imgIC)
        xI = np.int32(imgX)
        yI = np.int32(imgY)
        icf = np.int32(imgIC)
        ocf = np.int32(imgOC)
        xf = np.int32(fshape[0])
        yf = np.int32(fshape[1])
        f_displ = np.int32(filter.bshape[0])
        img_displ = np.int32(self.bshape[0])
        padding = np.int32(padding)


        cpu_earr = np.empty((imgX * imgY * imgOC, self.bshape[0]), dtype=self.device.DTYPE)
        cpu_earr = self.device.guardShapes(cpu_earr)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.device, resbuff, (imgX * imgY * imgOC, self.shape[1]), cpu_earr.shape)

        global_sizes = (int(imgX), int(imgY), int(self.shape[1]))
        local_sizes = None

        event = self.device.prg.conv2d(self.device.queue, global_sizes, local_sizes,
                                       cI, xI, yI,
                                       icf, ocf, xf, yf,
                                       f_displ, img_displ,
                                       padding,
                                       self.buffer, filter.buffer, res.buffer)
        cl.wait_for_events([event, ])
        return res


    def __mul__(self, other):
        assert self.shape[0] == other.shape[0]
        assert self.shape[1] == other.shape[1]
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.matdot(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, other.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def scale(self, scalar):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        scalar = self.device.DTYPE(scalar)
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.matscale(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, scalar, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def __matmul__(self, other):
        assert self.shape[0] == other.shape[0]
        M = np.int32(self.bshape[1])
        K = np.int32(self.bshape[0])
        N = np.int32(other.bshape[1])
        out_shape = (other.shape[1], self.shape[1])
        cpu_earr = np.empty((N, M), dtype=self.device.DTYPE)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.device, resbuff, out_shape, (N, M))
        global_sizes = (int(M/self.device.WPTM), int(N/self.device.WPTN))
        local_sizes = (int(self.device.TSM/self.device.WPTM), int(self.device.TSN/self.device.WPTN))
        event = self.device.prg.matmul(self.device.queue, global_sizes, local_sizes, M, N, K, self.buffer, other.buffer, res.buffer)
        cl.wait_for_events([event, ])
        return res

    def transpose(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        out_shape = (self.shape[1], self.shape[0])
        cpu_earr = np.empty((N, M), dtype=self.device.DTYPE)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.device, resbuff, out_shape, (N, M))
        global_sizes = (int(N), int(M))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.transpose(self.device.queue, global_sizes, local_sizes, N, M, self.buffer, res.buffer)
        cl.wait_for_events([event, ])
        return res

    def __add__(self, other):
        assert self.shape[0] == other.shape[0]
        assert self.shape[1] == other.shape[1]
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.matsum(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, other.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def __sub__(self, other):
        assert self.shape[0] == other.shape[0]
        assert self.shape[1] == other.shape[1]
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.matsubstract(self.device.queue, global_sizes, local_sizes, M, N, self.device.buffer, other.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def __mod__(self, other):
        assert self.shape[0] == other.shape[0]
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        K = np.int32(other.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.matpluscol(self.device.queue, global_sizes, local_sizes, M, N, K, self.buffer, other.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self


class Device(metaclass=Singleton):

    def __init__(self, devices: list, **kwargs):
        self.DTYPE = kwargs['DTYPE'] if 'DTYPE' in kwargs else np.float32
        assert self.DTYPE in DTYPES
        floatX = DTYPES[self.DTYPE]
        self.TSM = kwargs['TSM'] if 'TSM' in kwargs else 128
        self.TSN = kwargs['TSN'] if 'TSN' in kwargs else 128
        self.TSK = kwargs['TSK'] if 'TSK' in kwargs else 8
        self.TS = kwargs['TS'] if 'TS' in kwargs else 16
        self.WPTM = kwargs['WPTM'] if 'WPTM' in kwargs else 8
        self.WPTN = kwargs['WPTN'] if 'WPTN' in kwargs else 8
        self.ctx = cl.Context(devices)
        self.queue = cl.CommandQueue(self.ctx)
        options = "-DTSM={} -DTSN={} -DTSK={} -DWPTM={} -DWPTN={} -DfloatX={} -DTS={} -cl-mad-enable -cl-fast-relaxed-math".format(
            self.TSM, self.TSN, self.TSK, self.WPTM, self.WPTN, floatX, self.TS)
        self.prg = cl.Program(self.ctx, MAT_KERNELS).build(options)

    def guardShapes(self, M):
        if (M.shape[1] % self.TSN == 0) and (M.shape[0] % self.TSM == 0):
            return M.copy()
        else:
            nMx, nMy = M.shape[0], M.shape[1]
            if nMx % self.TSM != 0:
                nMx = (nMx // self.TSM + 1) * self.TSM
            if nMy % self.TSN != 0:
                nMy = (nMy // self.TSN + 1) * self.TSN
            nM = np.zeros((nMx, nMy), dtype=self.DTYPE)
            nM[:M.shape[0], :M.shape[1]] = M.copy()
            return nM.copy()

    def array(self, cpu_arr: np.ndarray):
        arr = self.guardShapes(cpu_arr.astype(self.DTYPE))
        buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
        return Array(self, buffer, cpu_arr.shape, arr.shape)

    def empty_array(self, shape):
        cpu_earr = np.empty(shape, dtype=self.DTYPE)
        earr = self.guardShapes(cpu_earr)
        buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=earr.nbytes)
        return Array(self, buffer, cpu_earr.shape, earr.shape)

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


