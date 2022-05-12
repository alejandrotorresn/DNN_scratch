#include "relu.h"

// Writing to a seperate output array isn'n needed as ReLU can be done in-place.
// However, for consistency, we should do it
void relu_forward_cpu(float *inp, float *out, int sz_out) {
    for (int i=0; i<sz_out; i++) {
        // Ternary operation: if 0 < inp[i], then output is in[i]. Otherwise, 0;
        out[i] = (0 < inp[i]) ? inp[i] : 0;
    }
}

ReLU_CPU::ReLU_CPU(int _sz_out) {
    sz_out = _sz_out;
}

void ReLU_CPU::forward(float *_inp, float *_out) {
    inp = _inp;
    out = _out;

    relu_forward_cpu(inp, out, sz_out);
}

void relu_backward_cpu(float *inp, float *out, int sz_out) {
    for (int i=0; i<sz_out; i++) {
        inp[i] = (0 < inp[i]) * out[i];
    }
}

void ReLU_CPU::backward() {
    relu_backward_cpu(inp, out, sz_out);
}