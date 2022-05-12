#include "linear.h"
#include "../utils/utils.h"

Linear_CPU::Linear_CPU(int _bs, int _n_in, int _n_out, float _lr) {
    bs = _bs;
    n_in = _n_in;
    n_out = _n_out;
    lr = _lr;

    sz_weights = n_in * n_out;
    sz_out = bs * n_out;

    weights = new float[sz_weights];
    bias = new float[n_out];

    kaiming_init(weights, n_in, n_out);
    init_zero(bias, n_out);
}

void linear_forward_cpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out) {
    int ind_inp, ind_weights, ind_out;
    for (int i=0; i<bs; i++) {
        for (int k=0; k<n_out; k++) {
            // Same as element out[i][k]
            ind_out = i*n_out + k;
            out[ind_out] = bias[k];

            // Vector multiplication
            for (int j=0; j<n_in; j++) {
                // Same as inp[i][j]
                ind_inp = i*n_in +j;
                // Sames as weights[i]][j]
                ind_weights = j*n_out + k;

                out[ind_out] += inp[ind_inp] * weights[ind_weights];
            }
        }
    }
}

void Linear_CPU::forward(float *_inp, float *_out) {
    inp = _inp;
    out = _out;

    linear_forward_cpu(inp, weights, bias, out, bs, n_in, n_out);
}

// Our backward passes calcultate the gradient of the loss with respeect to the input,
// assuming "out" contains the gradient of the loss with respect to the layer's out
// (i.e. the next later's input)
// The gradients are kept in "inp" (i.e. array "a" is replaced with the gradient of the loss )
// with respect to "a").
// Most librarues, such as PyTorrch, store the gradient in inp.grad to retain the original values
// as well, but that's beyond the scope of this series and we chouldn't need it anyway.
void linear_backward_cpu(float *inp, float *weights, float *out, int bs, int n_in, int n_out) {
    int ind_inp, ind_weights, ind_out;

    for (int i=0; i<bs; i++) {
        for (int k=0; k<n_out; k++) {
            // The formula for calculating the indices is the same as
            // it was in the forward pass.
            ind_out = i*n_out + k;

            for (int j=0; j<n_in; j++) {
                ind_inp = i*n_in + j;
                ind_weights = j*n_out + k;
                // "out" * the transpose of the weights
                inp[ind_inp] += weights[ind_weights]*out[ind_out];
            }
        }
    }
}

void Linear_CPU::backward() {
    // Zero out "inp" because the gradients are
    // added to it; they don't replace its elements.
    init_zero(inp, bs*n_in);

    // cp_weights is to clarified later
    linear_backward_cpu(inp, cp_weights, out, bs, n_in, n_out);

    delete[] cp_weights;
}

//linear_update_cpu assumes "out" contains the gradients of the loss with respect to the output of the layer,
// but "inp" es the original input of the layer. in other words, linear_update_cpu should be performed after
// we've performed the backwrd pass of the next layer but before doing so with this layer
void linear_update_cpu(float *inp, float *weights, float *bias, float *out, int bs, int n_in, int n_out, float lr) {
    int ind_inp, ind_weights, ind_out;

    for (int i=0; i<bs; i++) {
        for (int k=0; k<n_out; k++) {
            ind_out = i*n_out + k;
            bias[k] -= lr*out[ind_out];

            for (int j=0; j<n_in; j++) {
                ind_inp = i*n_in + j;
                ind_weights = j*n_out + k;

                weights[ind_weights] -= lr*inp[ind_inp]*out[ind_out];
            }
        }
    }
}

void Linear_CPU::update() {
    // Here's why we need a copy of the weights:
    // As already mentioned, we need to do linear_update_cpu
    // prior to linear_backward_cpu. The issue is that the former
    // updates the weights, but the latter requires the weights to be 
    // equal to what they were during the forward pass, so we can't
    // pass it the newly updated values. The solution is to store the
    // original weights in cp_weights, update the weights, and use
    // cp_weights for the backward pass. After we're done, we delete cp_weights
    // since we don't need it anymore.
    cp_weights = new float[n_in*n_out];
    set_eq(cp_weights, weights, sz_weights);

    linear_update_cpu(inp, weights, bias, out, bs, n_in, n_out, lr);
}