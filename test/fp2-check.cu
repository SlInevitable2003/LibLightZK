#include "alt_bn128_t.cuh"
#include "alt_bn128_init.cuh"

#include "xpu-vector.cuh"
using namespace alt_bn128;

__global__ void check(fp2_t *f)
{
    int tid = threadIdx.x;
    fp2_t a = f[(tid & 1) * 2 + 0], b = f[(tid & 1) * 2 + 1];
    
    if (tid == 0) a.print();
    __syncthreads();
    if (tid == 1) a.print();
    __syncthreads();

    if (tid == 0) b.print();
    __syncthreads();
    if (tid == 1) b.print();
    __syncthreads();

    fp2_t c;

    c = a + b;
    if (tid == 0) c.print();
    __syncthreads();
    if (tid == 1) c.print();
    __syncthreads();

    c = a - b;
    if (tid == 0) c.print();
    __syncthreads();
    if (tid == 1) c.print();
    __syncthreads();

    c = a * b;
    if (tid == 0) c.print();
    __syncthreads();
    if (tid == 1) c.print();
    __syncthreads();
}

int main(int argc, char *argv[])
{
    libff::init_alt_bn128_params();

    xpu::vector<libff::alt_bn128_Fq> f(4, xpu::mem_policy::cross_platform);
    f[0] = libff::alt_bn128_Fq::random_element(), f[1] = libff::alt_bn128_Fq::random_element();
    f[2] = libff::alt_bn128_Fq::random_element(), f[3] = libff::alt_bn128_Fq::random_element();
    
    f.store();
    check<<<1, 2>>>((fp2_t*)f.p());
    CUDA_DEBUG;

    libff::alt_bn128_Fq2 a(f[0], f[2]), b(f[1], f[3]);
    a.print(); b.print();
    (a + b).print();
    (a - b).print();
    (a * b).print();

    return 0;
}