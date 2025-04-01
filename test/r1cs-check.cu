#include "omp.h"

#include "timer.cuh"
#include "device-utils.cuh"
#include "xpu-spmat.cuh"
#include "alt_bn128_g1.cuh"
#include "alt_bn128_t.cuh"
using namespace std;

int main(int argc, char *argv[])
{
    libff::init_alt_bn128_params();

    xpu::spmat<libff::alt_bn128_Fr> A(xpu::application_scence::MatMul, xpu::matrix_type::A, 2);
    xpu::spmat<libff::alt_bn128_Fr> B(xpu::application_scence::MatMul, xpu::matrix_type::B, 2);
    xpu::spmat<libff::alt_bn128_Fr> C(xpu::application_scence::MatMul, xpu::matrix_type::C, 2);

    // A.print(); B.print(); C.print();

    return 0;
}