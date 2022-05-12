#include "relu.h"

__global__ void relu_forward_gpu(float *inp, float *out, int sz_out) {
    // Equivalent of the oterator "i" in the CPU code
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    // The reason we ensure the indez is smaller
    // than the size of the input is that the number of thrreads is usually
    // slightly larger than the size of the input (discussed in the previous article),
    // in which case we'd get a buffer overflow if we try to access elements corresponding to those extra threads.
    if (ind < sz_out) {
        // fmax is a CUDA function that returns the maximum of two floats
        // (0 and inp[ind] here, which is the definition of ReLU)
        // We could've used a ternary operation, but the CUDA math API is (here a tiny bit) more efficient
        out[ind] = fmaxf(0, inp[ind]);
    }
}

ReLU_GPU::ReLU_GPU(int _sz_out) {
    sz_out = _sz_out;
    n_blocks = (sz_out + block_size - 1) / block_size;
}

void ReLU_GPU::forward(float *_inp, float *_out) {
    inp = _inp;
    out = _out;
    // One-dimensional kernels can receive integers as the number of blocks & threads
    // No need for a dim3 object
    cudaDeviceSynchronize();
}

__global__ void relu_backward_gpu(float *inp, float *out, int sz_out) {
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out) {
        inp[ind] = (0 < inp[ind]) * out[ind];
    }
}

void ReLU_GPU::backward() {
    relu_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}