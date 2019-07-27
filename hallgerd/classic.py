import sklearn
import numpy as np
import pyopencl as cl
from tqdm import tqdm
from hallgerd.cl import *
from hallgerd.utils.math import sigmoid

os.environ['PYOPENCL_CTX'] = '0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


class LogisticRegressionCL:

    def __init__(self, C=0, lr=0.1, 
                 max_iter=1000, tol=0.1, 
                 init=np.zeros,
                 verbose=False):
        '''
        Linear classifier with gradient descend on OpenCL device
        Support only two classes: -1, 1
        :C : float
            Regularizatino term
        :lr : float
            Learning rate
        :max_iter : int
            Maximum iteration
        :tol : float
            Minimal vector weight difference for stop criterion
        :init : func(shape: tuple) -> np.ndarray
            Function for weight initialization
        :verbose : bool
            Verbose?
        '''
        self.C = C; self.lr = lr
        self.max_iter = max_iter; self.tol = tol
        self.init = init
        self.verbose = verbose

        self.w_h = None
        self.w_g = None
        
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError('No OpenCL platforms')
        else:
            p = platforms[0]
            devices = p.get_devices()
            if not devices:
                raise RuntimeError('No OpenCL devices')
            print('using ', devices[0].name)
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            self.prg = cl.Program(self.ctx, GD_CL_KERNELS).build()
    
    def fit(self, X, y):
        # Check and format X and y arrays
        X_h, y_h = sklearn.utils.check_X_y(X, y)
        pbar = tqdm(desc='fitting...', disable = not self.verbose)
        N = X_h.shape[0]
        X_h = X_h.T.astype('float32')
        y_h = y_h.astype('float32')
        # Init arrays
        self.w_h = self.init(shape=X.shape[1]).astype('float32')
        dw_h = np.empty_like(self.w_h)
        displ1_h = np.int64(X_h.shape[0])
        displ2_h = np.int64(X_h.shape[1])
        nworkers1 = X_h.shape[1]
        nworkers2 = X_h.shape[0]
        X_h = X_h.flatten('F')
        # And create cl buffers
        C_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float32(self.C))
        lr_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float32(self.lr))
        N_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))
        self.w_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.w_h)
        X_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_h)
        y_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y_h)
        R_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=X_h.nbytes)
        displ1_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=displ1_h)
        displ2_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=displ2_h)
        dw_g = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=dw_h.nbytes)
        # Update weights
        for i in range(self.max_iter):
            self.prg.wgrad(self.queue, (nworkers1,), None, self.w_g, X_g, y_g, displ1_g, R_g)
            self.prg.mreduction(self.queue, (nworkers2,), None, R_g, displ1_g, displ2_g, dw_g)
            self.prg.wupdate(self.queue, (nworkers2,), None, self.w_g, dw_g, C_g, lr_g, N_g)
            cl.enqueue_copy(self.queue, dw_h, dw_g)            
            vwnorm = np.linalg.norm(dw_h) / N
            pbar.set_description(desc='epoch: {}, wnorm: {}'.format(i, vwnorm))
            if vwnorm < self.tol:
                break
        return True

    def predict_proba(self, X):
        # Init arrays
        y_h = np.empty(X.shape[0], dtype='float32')
        X_h = X.T.astype('float32')
        displ = X_h.shape[0]
        nworkers = X_h.shape[1]
        X_h = X_h.flatten('F')
        # Create buffers
        X_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_h)
        y_g = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=y_h.nbytes)
        displ_g = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(displ))
        # Eval
        self.prg.vmmul(self.queue, (nworkers,), None, self.w_g, X_g, y_g, displ_g)
        cl.enqueue_copy(self.queue, y_h, y_g)
        return sigmoid(y_h)

    def predict(self, X):
        pred = self.predict_proba(X)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = -1
        return pred.astype('int')