#include <iostream>
#include "relu.h"
#include "../utils/utils.h"

int main() {
    // With ReLU, although we're mainly operating on bs*n_feats matrices,
    // we just treat our data like one-dimensional vectors, hence the need for just one dimension.
    int  bs;
    float *inp_cpu, *out_cpu;

    bs = random_int(128, 2048);

    inp_cpu = new float[bs];
    out_cpu = new float[bs];

    fill_array(inp_cpu, bs);

    ReLU_CPU relu_cpu(bs);

    for (int i=0; i<10; i++) {
        relu_cpu.forward(inp_cpu, out_cpu);
        std:: cout << relu_cpu.inp[i] << " ";
        relu_cpu.backward();
        std:: cout << relu_cpu.inp[i] << std::endl;
    }
}
