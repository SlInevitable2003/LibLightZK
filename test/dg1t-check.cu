#include "device-utils.cuh"
#include "xpu-vector.cuh"
#include "dalt_bn128_t.cuh"

#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"
using namespace std;

const size_t n = 128;

__global__ void add_test(alt_bn128::dg1_t *a, alt_bn128::dg1_t *b)
{
    alt_bn128::dg1_t aa, bb;
    aa.read_from(a + threadIdx.x / 8 * 8);
    bb.read_from(b + threadIdx.x / 8 * 8);
    aa.dbl();
    aa.write_to(a + threadIdx.x / 8 * 8);
}

int main(int argc, char *argv[])
{
    libff::init_alt_bn128_params();

    xpu::vector<libff::alt_bn128_G1> a(n, xpu::mem_policy::cross_platform), b(n, xpu::mem_policy::cross_platform), a_copy(n, xpu::mem_policy::host_only);
    for (size_t i = 0; i < n; i++) {
        a[i] = libff::alt_bn128_G1::random_element(), b[i] = libff::alt_bn128_G1::random_element();
        a[i].print(); b[i].print(); a_copy[i] = a[i];
    }
    a.store(), b.store();

    add_test<<<1, 8*n>>>((alt_bn128::dg1_t*)a.p(), (alt_bn128::dg1_t*)b.p());
    CUDA_DEBUG;
    a.load();
    for (size_t i = 0; i < n; i++) {
        a[i].print();
        (a_copy[i] + a_copy[i]).print();
        printf("%d: %s\n", i, (a[i] == (a_copy[i] + a_copy[i])) ? "Equal" : "Not equal");
    }

    return 0;
}