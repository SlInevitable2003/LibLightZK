#include <vector>

#include "omp.h"
#include "timer.cuh"
#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"
using namespace std;

const size_t n = 1 << 10;

int main(int argc, char *argv[])
{
    Timer timer;

    srand(time(0));
    libff::init_alt_bn128_params();

    vector<libff::alt_bn128_G1> ps(n);

    timer.start();
    #pragma omp parallel for
    for (size_t i = 0; i < n / 2; i++) ps[i] = libff::alt_bn128_G1::random_element();
    #pragma omp parallel for
    for (size_t i = 0; i < n / 2; i++) ps[i + n/2] = ps[rand() % (n/2)];
    timer.stop("Data prepare");

    libff::alt_bn128_G1 res1 = libff::alt_bn128_G1::zero(), res2 = libff::alt_bn128_G1::zero();

    timer.start();
    for (size_t i = 0; i < n; i++) res1 = res1 + ps[i];
    timer.stop("Serial Accumulation");

    timer.start();
    const int num_threads = omp_get_max_threads();
    vector<libff::alt_bn128_G1> local_acc(num_threads, libff::alt_bn128_G1::zero());
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n; i++) local_acc[tid] = local_acc[tid] + ps[i];
    }
    for (size_t i = 0; i < num_threads; i++) res2 = res2 + local_acc[i];
    timer.stop("Parallel Acuumulation");

    assert(res1 == res2);
    printf("Everything is OK!\n");

    vector<libff::alt_bn128_G2> p2s(n);

    timer.start();
    #pragma omp parallel for
    for (size_t i = 0; i < n / 2; i++) p2s[i] = libff::alt_bn128_G2::random_element();
    #pragma omp parallel for
    for (size_t i = 0; i < n / 2; i++) p2s[i + n/2] = p2s[rand() % (n/2)];
    timer.stop("Data prepare");

    libff::alt_bn128_G2 res21 = libff::alt_bn128_G2::zero(), res22 = libff::alt_bn128_G2::zero();

    timer.start();
    for (size_t i = 0; i < n; i++) res21 = res21 + p2s[i];
    timer.stop("Serial Accumulation");

    timer.start();
    vector<libff::alt_bn128_G2> local_acc2(num_threads, libff::alt_bn128_G2::zero());
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n; i++) local_acc2[tid] = local_acc2[tid] + p2s[i];
    }
    for (size_t i = 0; i < num_threads; i++) res22 = res22 + local_acc2[i];
    timer.stop("Parallel Acuumulation");

    assert(res21 == res22);
    printf("Everything is OK!\n");

    return 0;
}