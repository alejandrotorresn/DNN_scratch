#include "sequential.h"

// Note that "out" is there for consistency, and it is not used at all
// The utput is kept in last_layer -> out
void sequential_forward_cpu(float *inp, std::vector<Module*> layers, float *out) {
    int sz_out;
    float *curr_out;

    for (int i=0; i<layers.size(); i++) {
        Module *layer = layers[i];

        // if you're not familiar with -> in c++
        // please check this (https://stackoverflow.com/questions/1238613/what-is-the-difference-between-the-dot-operator-and-in-c) out
        // Basically, when foo is pointing to an object bar,
        // foo -> bar would be the equivalent to have a sz_out.
        sz_out = layer->sz_out;

        // "out" receives the output of the final layer.
        // For the intermediate layers, we need to create a container,
        // which is why requiere every layer to have a sz_out.
        curr_out = new float[sz_out];
        layer->forward(inp, curr_out);

        // the next layer's input is this layer's output
        inp = curr_out;
    }

    curr_out = new float[1];
    delete[] curr_out;
}

Sequential_CPU::Sequential_CPU(std::vector<Module*> _layers) {
    layers = _layers;
}

void Sequential_CPU::forward(float *inp, float *out) {
    sequential_forward_cpu(inp, layers, out);
}

void sequential_update_cpu(std::vector<Module*> layers) {
    // we start with the final layer and work ouur way backwards
    for (int i=layers.size()-1; 0<=i; i--) {
        Module *layer = layers[i];

        // For ReLU, update does nothing at all
        // It is onlye when the layer is a linear layer that it actually updates some parameters
        layer->update();
        layer->backward();
    }
}

void Sequential_CPU::update() {
    sequential_update_cpu(layers);
}