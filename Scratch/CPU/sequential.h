#ifndef SEQUENTIAL_CPU_H
#define SEQUENTIAL_CPU_H

#include <vector>

#include "../utils/module.h"

class Sequential_CPU: public Module {
    public:
        // "Layers" is a vector of pointers to Module-inherited objects like ReLU_CPU
        // Yet another reason for having a Module parent class is that
        // it's normally be tricky to have a vector of objects of different
        // types unless they all inherit the same class
        std::vector<Module*> layers;

        Sequential_CPU(std::vector<Module*> _layers);
        void forward(float *inp, float *out);
        // There is no backward, only update.
        // We'll see why in the following articles
        void update();
};

#endif