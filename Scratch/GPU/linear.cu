// __global__ means the fucntion, also called a kernel, should be run in parallel on the GPU, A.K.A the device
// The function is accesible by the CPU, A.K.A. the host

#include "linear.h"
#include "../utils/utils.h"

__global__ void linear_forward_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out) {
    int row = blockDim.x*blockIdx.x + threadIdx.x;
    int col = blockDim.y*blockIdx.y + threadIdx.y;

    if (( row < bs ) && ( col < n_out)) {
        ind_out = row*n_out + col;
        out[ind_out] = bias[col];

        for (int i=0; i<n_in; i++) {
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            out[ind_out] += inp[ind_inp]*weights[ind_weights];
        }
    }
}

Linear_GPU::Linear_GPU(int _bs, int _n_in, int _n_out, float _lr) {
    bs = _bs;
    n_in = _n_in;
    n_out = _n_out;
    lr = _lr;

    sz_weights = n_in*n_out;
    sz_out = bs*n_out;

    // The formula for each dimensional is the same
    n_block_rows = (bs + block_size - 1) / block_size;
    n_block_cols = (n_out + block_size -1) / block_size;

    // This is to allocate GPU Memory for our parameters.
    // It is the equivalent of new float[sz].
    // Parameters accesible throught both the CPU and the GPU
    // This is Unified Memory
    // The first argument is a reference to our array,
    // second is the number of bytes we need
    cudaMallocManaged(&weigts, sz_weights*sizeof(float));
    cudaMallocManaged(&bias, n_out*sizeof(float));

    // Parameter initializacion could be performed on the GPu for speed.
    // but CPU is fine since it is one-time
    kaiming_init(weights, n_in, n_out);
    init_zero(bias, n_out);
}

void Linear_GPU::forward(float *_inp, float *_out) {
    inp = _inp;
    out = _out;

    // The first argument refers to the number of blocks on the x-axis
    // The second argument referes to the number of blocks on the y-axis
    dim3 n_blocks(n_block_rows, n_block_cols);
    // The first argument refers to the number of the threads per block on the x-axis
    // The second argument refers to the number of threads per block on the y-axis
    dim3 n_threads(block_size, block_size);

    linear_forward_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out);

    // Wait for linear_forward_gpu to finish
    // Without cudaDeviceSynchronze(), the kernel would be launched,
    // but the CPU wouldn't wait for it to dinish and would be off to the next command,
    // meaning the forward pass may not be done by the time the next line of code is executed.
    cudaDeviceSynchronize();
}

__global__ void linear_backward_gpu(float *inp, float *weights, float *out, int bs, int n_in, int n_out) {
    int row = blockDim.x*blockIdx.x + threadIdx.x;
    int col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)) {
        ind_out = row*n_out + col;

        for (int i=0; i<n_in; i++) {
            ind_inp = row*ind_inp + i;
            ind_weight = i*n_out + col;
            // Using AtomicAdd because multiple threads will be accessing inp[ind_inp]
            atomicAdd(&inp[ind_inp], weights[ind_weights]*out[ind_out]);
        }
    }
}

void Linear_GPU::backward() {
    init_zero(inp, bs*n_in);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_backward_gpu<<<n_blocks, n_threads>>>(inp, cp_weights, out, bs, n_in, n_out);
    cudaDeviceSynchronize();
    
    cudaFree(cp_weights);
}

__global__ void linear_update_gpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out, float lr) {
    int row = blockDim.x*blockIdx.x + threadIdx.x;
    int col = blockDim.y*blockIdx.y + threadIdx.y;
    int ind_inp, ind_weights, ind_out;

    if ((row < bs) && (col < n_out)) {
        ind_out = row*n_out + col;
        atomicAdd(&bias[col], -lr*out[ind_out]);

        for (int i=0; i<n_in; i++) {
            ind_inp = row*n_in + i;
            ind_weights = i*n_out + col;

            atomicAdd(&weights[ind_weights], -lr*inp[ind_inp]*out[ind_out]);
        }
    }
}

void Linear_GPU::update() {
    cudaMallocManaged(&cp_weights, sz_weights*sizeof(float));
    set_eq(cp_weights, weights, sz_weights);

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_size, block_size);

    linear_update_gpu<<<n_blocks, n_threads>>>(inp, weights, bias, out, bs, n_in, n_out, lr);
    cudaDeviceSynchronize();
}