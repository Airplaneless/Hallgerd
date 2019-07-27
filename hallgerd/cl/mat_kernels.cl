//MATRIX OPERATIONS
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

__kernel void transpose(__global double * A,
                        __global double * AT,
                        __global const size_t * displ,
                        __global const size_t * displt)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t _displ = *displ;
    size_t _displt = *displt;
    AT[j + i * _displt] = A[i + j * _displ];
}

__kernel void sum(__global double * A, __global double * B)
{
    size_t i = get_global_id(0);
    double buff = A[i] + B[i];
    A[i] = buff;
}

__kernel void sum_col(__global double * A,
                      __global double * b,
                      __global const size_t * displ)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t _displ = *displ;
    double buff = A[j + i * _displ] + b[i];
    A[j + i * _displ] = buff;
}

__kernel void dot(__global double * A, __global double * B)
{
    size_t i = get_global_id(0);
    double buff = A[i] * B[i];
    A[i] = buff;
}

__kernel void scalar_dot(__global double * A, __global const double * s)
{
    size_t i = get_global_id(0);
    double scalar = * s;
    A[i] = scalar * A[i];
}
//ACTIVATIONS
__kernel void sigmoid(__global double * x)
{
    size_t i = get_global_id(0);
    double buff = 1 + exp(-x[i]);
    x[i] = 1 / buff;
}

__kernel void d_sigmoid(__global double * sx)
{
    size_t i = get_global_id(0);
    sx[i] = sx[i] * (1 - sx[i]);
}