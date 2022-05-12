#ifndef SEQUENTIAL_GPU_H
#define SEQUENTIAL_GPU_H

#include "../utils/module.h"

#include<vector>

class Sequential_GPU: public Module {
    public:
        std::vector<Module*> layers;

        Sequential_GPU(std::vector<Module*> _layers);
        void forward(float *inp, float *out);
        void update();
};

#endif