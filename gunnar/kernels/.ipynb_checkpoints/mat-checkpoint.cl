__kernel void matmul(__global const size_t * N,
                     __global const size_t * K,
                     __global double * A,
                     __global double * B,
                     __global double * C)
{

    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t _N = *N;
    size_t _K = *K;

    double res = 0.0;
    for (size_t k = 0; k < _K; ++k) {
        res += A[k + i * _K] * B[j + k * _N];
    }

    C[j + i * _N] = res;
}