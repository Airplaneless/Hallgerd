import logging
import numpy as np
import pyopencl as cl


SUPPORTED_ACTIVATIONS = ['sigmoid', 'relu', 'softmax']


class Dense:

    def __init__(self, in_shape, out_shape, activation='sigmoid'):
        assert activation in SUPPORTED_ACTIVATIONS
        self.activation = activation
        self.weight = np.random.randn(out_shape, in_shape).astype(np.float64) * np.sqrt(2 / in_shape)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.bias = np.zeros((out_shape, 1)).astype(np.float64)
        self.ctx = None
        self.queue = None
        self.prg = None
        self.input_cl = None
        self.output_cl = None
        self.weight_cl = None
        self.bias_cl = None
        self.dweight_cl = None
        self.dbias_cl = None
        self.error_cl = None
        self._batches = None

    def __connect_context__(self, ctx, queue, prg):
        self.ctx = ctx
        self.queue = queue
        self.prg = prg
        self.weight_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.weight)
        self.dweight_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.weight.nbytes)
        self.bias_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.bias)
        self.dbias_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.bias.nbytes)
        return True

    def __weight2cpu__(self):
        cl.enqueue_copy(self.queue, self.weight, self.weight_cl)
        cl.enqueue_copy(self.queue, self.bias, self.bias_cl)

    def __call__(self, input_cl, batches):
        # y = np.matmul(self.weight, x) + self.bias
        self.input_cl = input_cl
        self._batches = batches
        M = self.out_shape
        M_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(M))
        K = self.in_shape
        K_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(K))
        N = self._batches
        N_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))

        output_np = np.empty((M, N), dtype=np.float64)
        self.output_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=output_np.nbytes)

        self.prg.matmul(self.queue, (M, N), None, N_cl, K_cl, self.weight_cl, self.input_cl, self.output_cl)
        self.prg.sum_col(self.queue, (M, N), None, self.output_cl, self.bias_cl, N_cl)
        if self.activation == 'sigmoid':
            self.prg.sigmoid(self.queue, (M * N,), None, self.output_cl)
        if self.activation == 'relu':
            self.prg.relu(self.queue, (M * N,), None, self.output_cl)
        if self.activation == 'softmax':
            #TODO: do something with this
            buff_np = np.empty((M, N)).astype(np.float64)
            cl.enqueue_copy(self.queue, buff_np, self.output_cl)
            max = -np.max(buff_np)
            max_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(max))
            self.prg.scalar_sum(self.queue, (M * N,), None, self.output_cl, max_cl)
            self.prg.exp(self.queue, (M * N,), None, self.output_cl)
            v = np.empty(N, dtype=np.float64)
            v_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=v.nbytes)
            self.prg.sumreduce(self.queue, (M, N), None, self.output_cl, v_cl, M_cl, N_cl)
            self.prg.inverse(self.queue, (N,), None, v_cl)
            self.prg.dot2(self.queue, (M, N), None, self.output_cl, v_cl)
        return self.output_cl

    def backprop(self, error_cl, lr):
        lr_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(lr))
        # err = err * sigmoid_der(self.y)
        if self.activation == 'sigmoid':
            self.prg.d_sigmoid(self.queue, (self.out_shape * self._batches, ), None, self.output_cl)
        if self.activation == 'relu':
            self.prg.d_relu(self.queue, (self.out_shape * self._batches,), None, self.output_cl)
        if self.activation == 'softmax':
            self.prg.d_softmax(self.queue, (self.out_shape * self._batches,), None, self.output_cl)
        self.prg.dot(self.queue, (self.out_shape * self._batches, ), None, error_cl, self.output_cl)
        # self.weight += np.matmul(err, self.x.T) * lr
        x = np.empty((self.in_shape, self._batches), dtype=np.float64)
        x_t_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=x.nbytes)
        displ_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(self._batches))
        displt_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(self.in_shape))
        self.prg.transpose(self.queue, (self._batches, self.in_shape), None, self.input_cl, x_t_cl, displ_cl, displt_cl)
        M = self.out_shape
        K = self._batches
        K_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(K))
        N = self.in_shape
        N_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))
        self.prg.matmul(self.queue, (M, N), None, N_cl, K_cl, error_cl, x_t_cl, self.dweight_cl)
        self.prg.scalar_dot(self.queue, (M*N,), None, self.dweight_cl, lr_cl)
        self.prg.sum(self.queue, (M*N,), None, self.weight_cl, self.dweight_cl)
        N_cl.release()
        displt_cl.release()
        displ_cl.release()
        x_t_cl.release()
        # self.bias += np.matmul(err, np.ones((self.x.shape[1], 1))) * lr
        ones = np.ones((self._batches, 1)).astype(np.float64)
        ones_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ones)
        N = 1
        N_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))
        self.prg.matmul(self.queue, (M, N), None, N_cl, K_cl, error_cl, ones_cl, self.dbias_cl)
        self.prg.scalar_dot(self.queue, (M * N,), None, self.dbias_cl, lr_cl)
        self.prg.sum(self.queue, (M * N,), None, self.bias_cl, self.dbias_cl)
        K_cl.release()
        N_cl.release()
        ones_cl.release()
        # error = np.matmul(self.weight.T, err)
        weight_t = np.empty((self.in_shape, self.out_shape), dtype=np.float64)
        weight_t_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=weight_t.nbytes)
        displ_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(self.in_shape))
        displt_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(self.out_shape))
        self.prg.transpose(self.queue, (self.in_shape, self.out_shape), None, self.weight_cl, weight_t_cl, displ_cl, displt_cl)
        next_error = np.empty((self.in_shape, self._batches), dtype=np.float64)
        next_error_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=next_error.nbytes)
        M = self.in_shape
        K = self.out_shape
        K_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(K))
        N = self._batches
        N_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))
        self.prg.matmul(self.queue, (M, N), None, N_cl, K_cl, weight_t_cl, error_cl, next_error_cl)
        N_cl.release()
        K_cl.release()
        weight_t_cl.release()
        error_cl.release()
        displ_cl.release()
        displt_cl.release()
        return next_error_cl
