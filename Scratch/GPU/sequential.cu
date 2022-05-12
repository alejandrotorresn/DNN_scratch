#include "sequential.h"

void sequential_forward_gpu(float *inp, std::vector<Module*> layers, float *out) {
    int sz_out;
    float *curr_out;

    for (int i=0; i<layers.size(); i++) {
        Module *layer = layers[i];

        sz_out = layer->sz_out;

        // curr_out needs to be accesible through the GPU as well as the CPU.
        // so we use cudaMallocManaged
        cudaMallocManaged(&curr_out, sz_out*sizeof(float));
        layer->forward(inp, out);

        inp = curr_out;
    }

    cudaMallocManaged(&curr_out, sizeof(float));
    cudaFree(curr_out);
}

Sequential_GPU::Sequential_GPU(std::vector<Module*> _layers) {
    layers = _layers;
}

void Sequential_GPU::forward(float *inp, float *out) {
    sequential_forward_gpu(inp, layers, out);
}

// sequential_update_gpu is identical to sequential_update_cpu
// However, for completenes, we'll have two separate functions
void sequential_update_gpu(std::vector<Module *> layers) {
    for (int i=layers.size()-1; 0<=i; i--) {
        Module *layer = layers[i];

        layer->update();
        layer->backward();
    }
}

void Sequential_GPU::update() {
    sequential_update_gpu(layers);
}