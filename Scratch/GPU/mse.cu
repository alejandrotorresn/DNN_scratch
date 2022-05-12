#include "mse.h"

__global__ void mse_forward_gpu(float *inp, float *out, int sz_out) {
    int ind = blockDim.x*blockIdx.x + threadIdx.x;
    if (ind < sz_out) {
        // All the threads are going to be accessing out[sz_out]
        // which may cause issues if we simply do out[sz_out] += error.
        // atomicAdd is a thread-safe way to add a value to an address.
        // The first argument is the address, the second is the value to be added.
        // It is an in-place operation
        // powf is a CUDA function that raises the firts value to the power of the second one (i.e. the square difference).
        // fdividef is a CUDA function that divides the first argument by the second one (i.e. the average)
        atomicAdd( &out[sz_out], fdividef( powf( inp[ind]-out[ind], 2), sz_out ) );
    }
}

MSE_GPU::MSE_GPU(int _sz_out) {
    sz_out = _sz_out;
    n_blocks = (sz_out + block_size - 1) / block_size;
}

void MSE::forward(float *_inp, float *_out) {
    inp = _inp;
    out = _out;
}

void MSE_GPU::_forward(float *_inp, float *_out) {
    _out[sz_out] = 0.0f;
    mse_forward_gpu<<<n_blocks, block_size>>>(_inp, _out, sz_out);
    cudaDeviceSynchronize();
}

// Reminder: "inp" is the model's output, "out" is the target
__global__ void mse_backward_gpu(floar *inp, float *out, int sz_out) {
    int ind = blockdim.x * blockIdx.x + threadIdx.x;

    if (ind < sz_out) {
        // fdividef is a CUDA function that divides the first argument by the second one
        inp[ind] = fdividef(2*(inp[ind]-out[ind]), sz_out);
    }
}

void MSE_GPU::backward() {
    mse_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}