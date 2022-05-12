#ifndef RELU_CPU_H
#define RELU_CPU_H

#include "../utils/module.h"

class ReLU_CPU: public Module {
    public:
        // _sz_out is b*n_features
        // It's also equal to the size of the input
        // ReLU doesn't have any parameters to be updated
        ReLU_CPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void backward();
};

#endif