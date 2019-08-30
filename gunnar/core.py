import numpy as np
import pyopencl as cl

from functools import reduce

from gunnar.kernels import MAT_KERNELS
from hallgerd.losses import softmax

SUPPORTED_ACTIVATIONS = ['sigmoid', 'relu', 'softmax', 'linear']
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

    def im2col(self, batches, C, H, W, fh, fw, stride):
        batches = np.int32(batches);    C = np.int32(C)
        H = np.int32(H);    W = np.int32(W)
        nfh = (H - fh) // stride + 1
        nfw = (W - fw) // stride + 1
        M = np.int32(nfh * nfw * batches);  N = np.int32(C * fh * fw)

        cpu_earr = np.empty((M, N), dtype=self.device.DTYPE)
        cpu_earr = self.device.guardShapes(cpu_earr)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Array(self.device, resbuff, (M, N), cpu_earr.shape)

        global_sizes = (int(res.bshape[0]), int(res.bshape[0]))
        local_sizes = (int(self.device.TS), int(self.device.TS))

        event = self.device.prg.im2col(self.device.queue, global_sizes, local_sizes, self.buffer, res.buffer)
        cl.wait_for_events([event, ])
        return res

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

    def linear(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.linear(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, self.buffer)
        cl.wait_for_events([event, ])
        return self

    def dlinear(self):
        M = np.int32(self.bshape[0])
        N = np.int32(self.bshape[1])
        global_sizes = (int(M), int(N))
        local_sizes = (int(self.device.TS), int(self.device.TS))
        event = self.device.prg.dlinear(self.device.queue, global_sizes, local_sizes, M, N, self.buffer, self.buffer)
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


class Image(Array):
    
    def __init__(self, device, buffer, shape, bshape, image_shape, channels):
        super().__init__(device, buffer, shape, bshape)
        self.image_shape = image_shape
        self.channels = channels

    def crop(self, area):
        assert len(self.channels) == 2
        xcrop, ycrop = area
        imgX, imgY = self.image_shape
        imgOC, imgIC = self.channels
        xI = np.int32(imgX);    yI = np.int32(imgY)
        imgX = np.int32(xcrop[1] - xcrop[0])
        imgY = np.int32(ycrop[1] - ycrop[0])
        x1 = np.int32(xcrop[0]);    x2 = np.int32(xcrop[1])
        y1 = np.int32(ycrop[0]);    y2 = np.int32(ycrop[1])
        imgIC = np.int32(imgIC);  imgOC = np.int32(imgOC)

        cpu_earr = np.empty((imgIC * imgOC * imgX * imgY, 1), dtype=self.device.DTYPE)
        cpu_earr = self.device.guardShapes(cpu_earr)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Image(self.device, resbuff, (imgIC * imgOC * imgX * imgY, 1), cpu_earr.shape, (imgX, imgY), (imgOC, imgIC,))

        iI_displ = np.int32(self.bshape[1])
        oI_displ = np.int32(res.bshape[1])
        
        gs0 = (imgX // self.device.CTS + 1) * self.device.CTS
        gs1 = (imgY // self.device.CTS + 1) * self.device.CTS
        global_sizes = (int(gs0), int(gs1))
        local_sizes = (int(self.device.CTS), int(self.device.CTS))
        
        event = self.device.prg.filtercrop(self.device.queue, global_sizes, local_sizes, imgIC, imgOC, xI, yI, x1, x2, y1, y2, imgX, imgY, iI_displ, oI_displ, self.buffer, res.buffer)
        cl.wait_for_events([event, ])
        return res

    def dconv2d(self, other, area, padding=0):
        xcrop, ycrop = area
        oimgX = np.int32(xcrop[1] - xcrop[0])
        oimgY = np.int32(ycrop[1] - ycrop[0])
        x1 = np.int32(xcrop[0]);    x2 = np.int32(xcrop[1])
        y1 = np.int32(ycrop[0]);    y2 = np.int32(ycrop[1])
        assert len(self.channels) == 1
        assert len(other.channels) == 1
        imgX, imgY = self.image_shape
        imgOC, imgIC = self.channels[0], other.channels[0]
        
        assert self.shape[0] == imgX * imgY * imgOC
        assert other.shape[0] == imgX * imgY * imgIC
        assert self.shape[1] == other.shape[1]

        ciI = np.int32(imgIC);  coI = np.int32(imgOC)
        xI = np.int32(imgX);    yI = np.int32(imgY)
        icf = np.int32(imgIC);  ocf = np.int32(imgOC);

        iI_displ = np.int32(self.bshape[1])
        oI_displ = np.int32(other.bshape[1])
        padding = np.int32(padding)
        batches = np.int32(self.shape[1])

        outX = imgX + imgX - 1
        outY = imgY + imgY - 1

        cpu_earr = np.empty((imgIC * imgOC * oimgX * oimgY, 1), dtype=self.device.DTYPE)
        cpu_earr = self.device.guardShapes(cpu_earr)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Image(self.device, resbuff, (imgIC * imgOC * oimgX * oimgY, 1), cpu_earr.shape, (oimgX, oimgY), (imgOC, imgIC))

        gs0 = (oimgX // self.device.CTS + 1) * self.device.CTS
        gs1 = (oimgY // self.device.CTS + 1) * self.device.CTS
        global_sizes = (int(gs0), int(gs1))
        local_sizes = (int(self.device.CTS), int(self.device.CTS))

        event = self.device.prg.dconv2d(self.device.queue, global_sizes, local_sizes, ciI, coI, xI, yI, icf, ocf, xI, yI, oimgX, oimgY, iI_displ, oI_displ, padding, batches, x1, x2, y1, y2, self.buffer, other.buffer, res.buffer)
        cl.wait_for_events([event, ])
        return res

    def conv2d(self, filter, padding=0, reverse=False):
        assert len(self.channels) == 1
        assert len(filter.channels) == 2
        
        imgX, imgY = self.image_shape
        if reverse:
            imgIC, imgOC = filter.channels
        else:
            imgOC, imgIC = filter.channels
        fshape = filter.image_shape

        assert self.shape[0] == imgX * imgY * imgIC
        assert filter.shape[0] == fshape[0] * fshape[1] * imgIC * imgOC
        assert filter.shape[1] == 1
        assert filter.image_shape[0] <= self.device.CTS / self.device.IBS
        assert filter.image_shape[1] <= self.device.CTS / self.device.IBS

        cI = np.int32(imgIC);   xI = np.int32(imgX);    yI = np.int32(imgY)
        icf = np.int32(imgIC);  ocf = np.int32(imgOC)
        xf = np.int32(fshape[0]);   yf = np.int32(fshape[1])
        f_displ = np.int32(filter.bshape[1])
        img_displ = np.int32(self.bshape[1])
        padding = np.int32(padding)
        batches = np.int32(self.shape[1])

        cpu_earr = np.empty((imgX * imgY * imgOC, self.shape[1]), dtype=self.device.DTYPE)
        cpu_earr = self.device.guardShapes(cpu_earr)
        resbuff = cl.Buffer(self.device.ctx, cl.mem_flags.READ_WRITE, size=cpu_earr.nbytes)
        res = Image(self.device, resbuff, (imgX * imgY * imgOC, self.shape[1]), cpu_earr.shape, (imgX, imgY), (imgOC,))

        gs0 = (imgX // self.device.CTS + 1) * self.device.CTS
        gs1 = (imgY // self.device.CTS + 1) * self.device.CTS
        gs2 = (batches // self.device.CTS + 1) * self.device.CTS
        global_sizes = (int(gs0), int(gs1), int(gs2))
        local_sizes = (int(self.device.CTS/self.device.IBS), int(self.device.CTS/self.device.IBS), int(self.device.IBS * 2))
        
        event = self.device.prg.conv2d(self.device.queue, global_sizes, local_sizes, cI, xI, yI, icf, ocf, xf, yf, f_displ, img_displ, padding, batches, self.buffer, filter.buffer, res.buffer)
        cl.wait_for_events([event, ])
        return res


class Device(metaclass=Singleton):

    def __init__(self, devices: list, **kwargs):
        self.DTYPE = kwargs['DTYPE'] if 'DTYPE' in kwargs else np.float32
        assert self.DTYPE in DTYPES
        floatX = DTYPES[self.DTYPE]
        self.TSM = kwargs['TSM'] if 'TSM' in kwargs else 128
        self.TSN = kwargs['TSN'] if 'TSN' in kwargs else 128
        self.TSK = kwargs['TSK'] if 'TSK' in kwargs else 8
        self.CTS = kwargs['CTS'] if 'CTS' in kwargs else 16
        self.TS = kwargs['TS'] if 'TS' in kwargs else 16
        self.IBS = kwargs['IBS'] if 'IBS' in kwargs else 4
        self.WPTM = kwargs['WPTM'] if 'WPTM' in kwargs else 8
        self.WPTN = kwargs['WPTN'] if 'WPTN' in kwargs else 8
        self.ctx = cl.Context(devices)
        self.queue = cl.CommandQueue(self.ctx)
        options = "-DTSM={} -DTSN={} -DTSK={} -DWPTM={} -DWPTN={} -DfloatX={} -DTS={} -DCTS={} -DIBS={} -cl-mad-enable -cl-fast-relaxed-math".format(self.TSM, self.TSN, self.TSK, self.WPTM, self.WPTN, floatX, self.TS, self.CTS, self.IBS)
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

    def image(self, cpu_arr: np.ndarray, shape, channels):
        arr = self.guardShapes(cpu_arr.astype(self.DTYPE))
        buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
        return Image(self, buffer, cpu_arr.shape, arr.shape, shape, channels)

    def empty_array(self, shape):
        cpu_earr = np.empty(shape, dtype=self.DTYPE)
        earr = self.guardShapes(cpu_earr)
        buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=earr.nbytes)
        return Array(self, buffer, cpu_earr.shape, earr.shape)

    def empty_image(self, shape, ishape, channels):
        cpu_earr = np.empty(shape, dtype=self.DTYPE)
        earr = self.guardShapes(cpu_earr)
        buffer = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=earr.nbytes)
        return Image(self, buffer, cpu_earr.shape, earr.shape, ishape, channels)

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


