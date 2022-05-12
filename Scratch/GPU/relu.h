#ifndef RELU_GPU_H
#define RELU_GPU_H

#include "../utils/module.h"

class ReLU_GPU: public Module {
    public:
        // Unlike the linear layer, where we had 2D blocks,
        // here there is only one axis, so there needs to be
        // just one variable addressing the total number of blocks
        int n_blocks;

        ReLU_GPU(int sz_out);
        void forward(float *_inp, float *_out);
        void backward();
};

#endif