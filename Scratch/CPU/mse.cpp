#include "mse.h"

// For consistency, we're not going to return the MSE.
// Instead, it will be stored in out[sz_out]
void mse_forward_cpu(float *inp, float *out, int sz_out) {
    for (int i=0; i<sz_out; i++) {
        // (inp[i] - out[i])*(inp[i] - out[i]) is the squared difference
        // We divide by sz_out to get the average. We could've added
        // all the squared differences and divided by sz_out only once
        // at the end, but this is easier to implement in CUDA.
        out[sz_out] += (inp[i] - out[i]) * (inp[i] - out[i]) / sz_out;
    }
}

MSE_CPU::MSE_CPU(int _sz_out) {
    sz_out = _sz_out;
}

// Since we actually don;t need the loss, we can have
// a dummy function "forward" that simple stores the input and output
// for backpropagation. We might use it if performance is very, very
// important, and we don't to waste time calculating the loss.
void MSE_CPU::forward(float *_inp, float *_out) {
    inp = _inp;
    out = _out;
}

// In case we do decide to compute the MSE,
// we use "_forward". Note that it doesn't store
// the input and output, meaning prior to backpropagation,
// we always need to call "forward".
void MSE_CPU::_forward(float *_inp, float *_out) {
    // Zero out _out[sz_out] sice we don't replace it with the MSE;
    // we add the MSE to it.
    _out[sz_out] = 0.0f;

    mse_forward_cpu(_inp, _out, sz_out);
}

// Reminder: "inp" is the model's output, "out" is the target
void mse_backward_cpu(float *inp, float *out, int sz_out) {
    for (int i=0; i<sz_out; i++) {
        // MSE: ((inp[i]-out[i])**2) / sz_out
        // Equato to: (inp[i]**2 - 2*inp[i]*out[i] + out[i]**2) / sz_out
        // Derivate of the value inside the parenthesis with respect to inp[i]: 
        //      2*inp[i] (power rule) - 2*out[i] (the multiple rule)
        // Equal to: 2*(inp[i] - out[i])
        // Derivative of the whole thing:
        inp[i] = 2*(inp[i]-out[i])/sz_out;
    }
}

void MSE_CPU::backward() {
    mse_backward_cpu(inp, out, sz_out);
}