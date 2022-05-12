#ifndef MSE_CPU_H
#define MSE_CPU_H

#include "../utils/module.h"

class MSE_CPU: public Module {
    public:
        // "inp" referes to the model's predictions while
        // "out" referes to the target values
        float *inp, *out;

        MSE_CPU(int _sz_out);
        // The difference between "forward" and "_forward" will be clarified
        void forward(float *_inp, float *_out);
        void _forward(float *_inp, float *_out);
        void backward();
};

#endif