#include "omp.h"

#include "timer.cuh"
#include "device-utils.cuh"
#include "xpu-vector.cuh"
#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"
#include "alt_bn128_t.cuh"
using namespace std;

const size_t n = 1 << 15;

#define BLK_SIZ 512

__global__ void accumulate_g1(alt_bn128::g1_t *ps, const size_t n)
{
    int tid = threadIdx.x;
    __shared__ alt_bn128::g1_t shmem[BLK_SIZ];
    shmem[tid].zero();
    __syncthreads();

    for (int i = tid; i < n; i += BLK_SIZ) shmem[tid].dadd(ps[i]);
    __syncthreads();

    for (int i = 1; i < BLK_SIZ; i <<= 1) {
        alt_bn128::g1_t incr = shmem[tid ^ i];
        __syncthreads();
        shmem[tid].dadd(incr);
        __syncthreads();
    }

    if (tid == 0) ps[0] = shmem[0];
}

__global__ __launch_bounds__(512, 1) void accumulate_g2(alt_bn128::g2_t *ps, const size_t n)
{
    int tid = threadIdx.x;
    __shared__ alt_bn128::g2_t shmem[BLK_SIZ];
    shmem[tid].zero();
    __syncthreads();

    alt_bn128::g2_t incr;
    for (int i = tid; i < 2 * n; i += BLK_SIZ) {
        incr.read_from((alt_bn128::fp_t*)(ps + (i&(~1))));
        shmem[tid].dadd(incr);
    }
    __syncthreads();

    for (int i = 2; i < BLK_SIZ; i <<= 1) {
        alt_bn128::g2_t incr = shmem[tid ^ i];
        __syncthreads();
        shmem[tid].dadd(incr);
        __syncthreads();
    }

    if (tid <= 1) shmem[tid].write_to((alt_bn128::fp_t*)ps);
}

int main(int argc, char *argv[])
{
    Timer timer;
    libff::init_alt_bn128_params();

    xpu::vector<libff::alt_bn128_G1> ps(n, xpu::mem_policy::cross_platform);

    timer.start();
    #pragma parallel for
    for (size_t i = 0; i < n; i++) ps[i] = libff::alt_bn128_G1::random_element();
    timer.stop("Data prepare");

    ps.store();
    CUDA_DEBUG;
    timer.start();
    accumulate_g1<<<1, BLK_SIZ>>>((alt_bn128::g1_t *)ps.p(), n);
    CUDA_DEBUG;
    timer.stop("CUDA Sum");

    timer.start();
    const int num_threads = omp_get_max_threads();
    vector<libff::alt_bn128_G1> local_acc(num_threads, libff::alt_bn128_G1::zero());
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n; i++) local_acc[tid] = local_acc[tid] + ps[i];
    }
    for (size_t i = 1; i < num_threads; i++) local_acc[0] = local_acc[0] + local_acc[i];
    timer.stop("Parallel Acuumulation");

    ps.load(1);
    assert(ps[0] == local_acc[0]);
    printf("Everything is OK!\n");

    xpu::vector<libff::alt_bn128_G2> p2s(n, xpu::mem_policy::cross_platform);
    timer.start();
    #pragma parallel for
    for (size_t i = 0; i < n; i++) p2s[i] = libff::alt_bn128_G2::random_element();
    timer.stop("Data prepare");

    p2s.store();
    CUDA_DEBUG;
    timer.start();
    accumulate_g2<<<1, BLK_SIZ>>>((alt_bn128::g2_t *)p2s.p(), n);
    CUDA_DEBUG;
    timer.stop("CUDA Sum");

    timer.start();
    vector<libff::alt_bn128_G2> local_acc2(num_threads, libff::alt_bn128_G2::zero());
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n; i++) local_acc2[tid] = local_acc2[tid] + p2s[i];
    }
    for (size_t i = 1; i < num_threads; i++) local_acc2[0] = local_acc2[0] + local_acc2[i];
    timer.stop("Parallel Acuumulation");

    p2s.load(1);
    assert(p2s[0] == local_acc2[0]);
    printf("Everything is OK!\n");

    return 0;
}