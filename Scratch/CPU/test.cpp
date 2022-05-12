#include <iostream>

#include "linear.h"
#include "../utils/utils.h"

int main () {
    int bs, n_in, n_out;
    int sz_in, sz_weights, sz_out;
    float *inp_cpu, *out_cpu;
    
    bs = random_int(32, 256);
    n_in = random_int(32, 64);
    n_out = random_int(1,32);

    sz_in = bs * n_in;
    sz_weights = n_in * n_out;
    sz_out = bs * n_out;

    inp_cpu = new float[sz_in];
    out_cpu = new float[sz_out];

    fill_array(inp_cpu, sz_in);

    Linear_CPU lin_cpu(bs, n_in, n_out);
    std::cout<<"Weight 1 = " << lin_cpu.weights[1] << std::endl;
    lin_cpu.forward(inp_cpu, out_cpu);
    lin_cpu.update();
    lin_cpu.backward();
    std::cout<<"Weight 1 = " << lin_cpu.weights[1] << std::endl;

    return 0;
}