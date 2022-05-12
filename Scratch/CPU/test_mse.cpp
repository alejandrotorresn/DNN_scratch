#include <iostream>
#include "mse.h"
#include "../utils/utils.h"

int main() {
    int bs;
    float *inp_cpu, *out_cpu;

    bs =random_int(32, 2048);
    inp_cpu = new float[bs];
    out_cpu = new float[bs];

    fill_array(inp_cpu, bs);
    fill_array(out_cpu, bs + 1);

    MSE_CPU mse_cpu(bs);

    std:: cout << inp_cpu[1] << std:: endl;
    std:: cout << out_cpu[1] << std:: endl;

    mse_cpu.forward(inp_cpu, out_cpu);
    mse_cpu._forward(inp_cpu, out_cpu);
    
    std:: cout << mse_cpu.inp[1] << " ";
    std:: cout << mse_cpu.out[1] << std::endl;

    mse_cpu.backward();

    std:: cout << mse_cpu.inp[1] << " ";
    std:: cout << mse_cpu.out[1] << std::endl;

    return 0;
}