__kernel void vmmul(__global const float * v,
                    __global const float * X,
                    __global float * r,
                    __global const size_t * displ)
{
    size_t i = get_global_id(0);
    size_t _displ = *displ;
    __local float lv[1024*8];

    event_t ecopy = async_work_group_copy(lv, v, 1024*8, 0);
    wait_group_events(1, &ecopy);
    
    float val = 0;
    for (size_t j = 0; j < _displ; ++j) {
        val += X[i * _displ + j] * lv[j];
    }
    r[i] = val;
}


__kernel void wgrad(__global float * vweight, 
                    __global const float * mX,
                    __global const float * vy,
                    __global const size_t * displ,
                    __global float * res)
{
    size_t i = get_global_id(0);
    size_t _displ = *displ;
    __local float lvweight[1024*8];

    event_t ecopy = async_work_group_copy(lvweight, vweight, 1024*8, 0);
    wait_group_events(1, &ecopy);
    // Eval y_pred vector
    float vyp = 0;
    for (size_t j = 0; j < _displ; ++j) {
        vyp += mX[i * _displ + j] * lvweight[j];
    }
    // Eval k vector
    float k = 1 - pow((1 + exp(-vyp * vy[i])), -1);
    // Eval res matrix
    for (size_t j = 0; j < _displ; ++j) {
        res[i * _displ + j] = vy[i] * k * mX[i * _displ + j];
    }    
}

__kernel void mreduction(__global float * mX,
                         __global const size_t * displx,
                         __global const size_t * disply,
                         __global float * vres) 
{
    size_t i = get_global_id(0);
    size_t _displx = *displx;
    size_t _disply = *disply;
    float value = 0;
    for (size_t j = 0; j < _disply; ++j) {
        value += mX[i + j * _displx ];
    }
    vres[i] = value;
}

__kernel void wupdate(__global float * vweight, 
                      __global float * vgrad,
                      __global const float * C,
                      __global const float * lr,
                      __global const size_t * N)
{
    size_t i = get_global_id(0);
    float _C = *C;
    float _lr = *lr;
    size_t _N = *N;
    vweight[i] = (1 - _lr * _C) * vweight[i] + (_lr / _N) * vgrad[i];
}