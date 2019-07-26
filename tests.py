import unittest
import warnings

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from hallgerd.classic import *


os.environ['PYOPENCL_CTX'] = '0'


class TestCLkernels(unittest.TestCase):
    
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
