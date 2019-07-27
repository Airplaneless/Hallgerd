import unittest
import warnings

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from hallgerd.classic import *


os.environ['PYOPENCL_CTX'] = '0'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TestCLkernels(unittest.TestCase):

    def test_softmax(self):

        def softmax(x):
            return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, MAT_CL_KERNELS).build()

        A = np.random.random((1000, 300)).astype(np.float64) + 30000
        rA = np.empty_like(A)
        A_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)

        res = softmax(A)
        M = A.shape[0]
        M_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(M))
        N = A.shape[1]
        N_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))
        buff_np = np.empty((M, N)).astype(np.float64)
        cl.enqueue_copy(queue, buff_np, A_cl)
        max = -np.max(buff_np)
        max_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(max))
        prg.scalar_sum(queue, (M * N,), None, A_cl, max_cl)
        prg.exp(queue, (M * N,), None, A_cl)
        v = np.empty(N, dtype=np.float64)
        v_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=v.nbytes)
        prg.sumreduce(queue, (M, N), None, A_cl, v_cl, M_cl, N_cl)
        prg.inverse(queue, (N,), None, v_cl)
        prg.dot2(queue, (M, N), None, A_cl, v_cl)
        cl.enqueue_copy(queue, rA, A_cl)
        self.assertGreater(1e-2, np.linalg.norm(res - rA), msg='wrong softmax')

    def test_relu(self):

        def relu(x):
            return np.maximum(0, x)

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, MAT_CL_KERNELS).build()

        A = np.random.random((1000, 30)).astype(np.float64)
        rA = np.empty_like(A)
        A_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)

        res = relu(A)
        prg.relu(queue, A.shape, None, A_cl)
        cl.enqueue_copy(queue, rA, A_cl)
        self.assertGreater(1e-2, np.linalg.norm(res - rA), msg='wrong relu')

    def test_dot(self):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, MAT_CL_KERNELS).build()

        A = np.random.random((1000, 30)).astype(np.float64)
        A_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        v = np.random.random(30).astype(np.float64)
        v_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=v)
        r = np.empty_like(A)
        r_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=A.nbytes)

        res = A * v
        prg.dot2(queue, A.shape, None, A_cl, v_cl)
        cl.enqueue_copy(queue, r, A_cl)
        self.assertGreater(1e-2, np.linalg.norm(res - r), msg='matrix sum reduce wrong')

    def test_sumreduce(self):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, MAT_CL_KERNELS).build()

        A = np.random.random((2000, 4)).astype(np.float64)
        A_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        v = np.empty((1, 4), dtype=np.float64)
        v_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=v.nbytes)
        ax0_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(A.shape[0]))
        ax1_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(A.shape[1]))

        res = np.sum(A, axis=0)

        prg.sumreduce(queue, A.shape, None, A_cl, v_cl, ax0_cl, ax1_cl)
        cl.enqueue_copy(queue, v, v_cl)
        self.assertGreater(1e-2, np.linalg.norm(res - v), msg='matrix sum reduce wrong')

    def test_matmul(self):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, MAT_CL_KERNELS).build()

        A = np.random.randn(1000, 2000).astype(np.float64)
        A_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B = np.random.randn(2000, 3000).astype(np.float64)
        B_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C = np.empty((A.shape[0], B.shape[1]), dtype=np.float64)
        C_cl = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=C.nbytes)
        assert A.shape[1] == B.shape[0]
        M = A.shape[0]
        K = B.shape[0]
        K_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(K))
        N = B.shape[1]
        N_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))

        res = np.matmul(A, B)

        prg.matmul(queue, (M, N), None, N_cl, K_cl, A_cl, B_cl, C_cl)
        cl.enqueue_copy(queue, C, C_cl)
        self.assertGreater(1e-2, np.linalg.norm(res - C), msg='matrix multiplication wrong')

    def test_matmul_and_sum_and_sigmoid_and_dot(self):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, MAT_CL_KERNELS).build()

        lr = 1e-1
        lr_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.float64(lr))
        A = np.random.randn(1000, 2000).astype(np.float64)
        A_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B = np.random.randn(2000, 3000).astype(np.float64)
        B_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C = np.empty((A.shape[0], B.shape[1]), dtype=np.float64)
        C_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=C.nbytes)
        D = np.random.randn(A.shape[0], B.shape[1]).astype(np.float64)
        D_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=D)
        assert A.shape[1] == B.shape[0]
        M = A.shape[0]
        K = B.shape[0]
        K_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(K))
        N = B.shape[1]
        N_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))

        res = sigmoid(np.matmul(A, B) + D) * lr

        prg.matmul(queue, (M, N), None, N_cl, K_cl, A_cl, B_cl, C_cl)
        prg.sum(queue, (M*N, ), None, C_cl, D_cl)
        prg.sigmoid(queue, (M*N, ), None, C_cl)
        prg.scalar_dot(queue, (M * N,), None, C_cl, lr_cl)
        cl.enqueue_copy(queue, C, C_cl)
        self.assertGreater(1e-2, np.linalg.norm(res - C), msg='matrix multiplication and sum wrong')

    def test_sum_col(self):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, MAT_CL_KERNELS).build()

        A = np.random.randn(1000, 2000).astype(np.float64)
        A_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B = np.random.randn(2000, 3000).astype(np.float64)
        B_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C = np.empty((A.shape[0], B.shape[1]), dtype=np.float64)
        C_cl = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=C.nbytes)
        v = np.random.randn(1000, 1).astype(np.float64)
        v_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v)
        displ_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(3000))
        assert A.shape[1] == B.shape[0]
        M = A.shape[0]
        K = B.shape[0]
        K_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(K))
        N = B.shape[1]
        N_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(N))

        res = np.matmul(A, B) + v

        prg.matmul(queue, (M, N), None, N_cl, K_cl, A_cl, B_cl, C_cl)
        prg.sum_col(queue, (M, N), None, C_cl, v_cl, displ_cl)
        cl.enqueue_copy(queue, C, C_cl)
        self.assertGreater(1e-2, np.linalg.norm(res - C), msg='sum col wrong')

    def test_transpose(self):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, MAT_CL_KERNELS).build()

        x = np.random.randn(1000, 2000).astype(np.float64)
        x_t = np.empty((2000, 1000), dtype=np.float64)
        x_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        x_t_cl = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=x.nbytes)
        displ_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(2000))
        displt_cl = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(1000))
        prg.transpose(queue, (2000, 1000), None, x_cl, x_t_cl, displ_cl, displt_cl)
        cl.enqueue_copy(queue, x_t, x_t_cl)
        self.assertGreater(1e-2, np.linalg.norm(x_t - x.T), msg='matrix transpose wrong')


    def test_vmmul(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            prg = cl.Program(ctx, GD_CL_KERNELS).build()

            w_h = np.random.random(200).astype(np.float32)
            X_h = np.random.random((200, 109000)).astype(np.float32)
            r_h = np.empty(109000, dtype=np.float32)

            rr = np.dot(w_h, X_h)

            X_size_h = X_h.shape[0]
            nworkers = X_h.shape[1]
            X_h = X_h.flatten('F')

            w_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_h)
            X_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_h)
            r_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=r_h.nbytes)
            X_size_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.int64(X_size_h))

            prg.vmmul(queue, (nworkers,), None, w_g, X_g, r_g, X_size_g)
            cl.enqueue_copy(queue, r_h, r_g)
            self.assertGreater(1e-2, np.linalg.norm(r_h - rr), msg='vector-matrix product wrong')

    def test_log_grad(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            prg = cl.Program(ctx, GD_CL_KERNELS).build()

            w_h = np.random.random(200).astype(np.float32)
            X_h = np.random.random((109000, 200)).astype(np.float32)
            y_h = np.random.random(109000).astype(np.float32)
            R_h = np.empty_like(X_h.T)
            dw_h = np.empty_like(w_h)

            mm = (X_h.T * y_h) * (1 - 1 / (1 + np.exp(-np.dot(X_h, w_h)*y_h)))
            res = np.sum(mm, axis=1)
            X_h = X_h.T

            displ1_h = np.int64(X_h.shape[0])
            displ2_h = np.int64(X_h.shape[1])
            nworkers1 = X_h.shape[1]
            nworkers2 = X_h.shape[0]
            # print(displ1_h, nworkers1)
            X_h = X_h.flatten('F')

            w_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_h)
            X_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_h)
            y_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y_h)
            R_g = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=R_h.nbytes)
            displ1_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=displ1_h)
            displ2_g = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=displ2_h)
            dw_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=dw_h.nbytes)

            prg.wgrad(queue, (nworkers1,), None, w_g, X_g, y_g, displ1_g, R_g)
            prg.mreduction(queue, (nworkers2,), None, R_g, displ1_g, displ2_g, dw_g)

            cl.enqueue_copy(queue, dw_h, dw_g)
            queue.finish()
            
            self.assertGreater(1e-2, np.linalg.norm(res - dw_h)) 


class TestModels(unittest.TestCase):
    
    def test_logreg_classifier(self):
        X, y = make_classification(n_samples=1000, random_state=42)
        y[y == 0] = -1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        clf = LogisticRegressionCL(verbose=False)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        self.assertGreater(score, 0.7, msg='LogReg underfit')
        

if __name__ == "__main__":
    unittest.main()
