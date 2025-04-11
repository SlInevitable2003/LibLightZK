#include <typeinfo>
#include <iostream>
#include "device-utils.cuh"
#include "timer.cuh"
#include "groth16-operator.cuh"
#include "alt_bn128_t.cuh"
#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"
#include "omp.h"

#include "alt_bn128_init.cuh"

const size_t n = 1 << 20;

const size_t win_siz = 11;
const size_t win_cnt = (256 + win_siz - 1) / win_siz;
const size_t bucket_cnt = 1 << win_siz;

int main(int argc, char *argv[])
{
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // std::cout << "Device name: " << prop.name << std::endl;
    // std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    // std::cout << "Streaming Multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;

    Timer timer;

    libff::init_alt_bn128_params();

    xpu::vector<libff::alt_bn128_G1> dev_points(win_cnt * n, xpu::mem_policy::cross_platform);  //device变量
    //xpu::vector<libff::alt_bn128_G1> check_points(n);
    xpu::vector<libff::alt_bn128_Fr> dev_scalars(n, xpu::mem_policy::cross_platform);    //混合变量

    timer.start();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_scalars[i] = libff::alt_bn128_Fr::random_element();
    dev_scalars.store();

    (fix_base_multi_scalar_multiplication_g1<alt_bn128::fr_t, alt_bn128::g1_t>)<<<160, 510>>>((alt_bn128::g1_t*)dev_points.p(), (alt_bn128::fr_t*)dev_scalars.p(), n);

    dev_points.load();

    // for(int i=0;i<n;i++){
    //     check_points[i] = libff::alt_bn128_G1::G1_one;
    //     check_points[i] = dev_scalars[i] * check_points[i];
    //     assert(check_points[i] == dev_points[i]);
        
    // }
    (pre_comp_g1<alt_bn128::g1_t>)<<<160, 1024>>>(n, (alt_bn128::g1_t*)dev_points.p(), win_siz, win_cnt);
    CUDA_DEBUG;
    
    timer.stop();

    
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_scalars[i] = libff::alt_bn128_Fr::random_element();
    dev_scalars.store();

    CUDA_DEBUG;
    timer.start();
    multi_scalar_multiplication_g1<alt_bn128::fr_t, alt_bn128::g1_t>(n, (alt_bn128::fr_t*)dev_scalars.p(), (alt_bn128::g1_t*)dev_points.p(), (alt_bn128::g1_t*)dev_points.p() + n, win_siz, win_cnt, bucket_cnt, n);
    CUDA_DEBUG;
    timer.stop();


    xpu::vector<libff::alt_bn128_G1> points(n+1);
    points.load(dev_points.p(), n+1);
    const int num_threads = omp_get_max_threads();
    timer.start();
    std::vector<libff::alt_bn128_G1> local_acc(num_threads, libff::alt_bn128_G1::zero());
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n; i++) local_acc[tid] = local_acc[tid] + dev_scalars[i] * points[i];
    }
    for (size_t i = 1; i < num_threads; i++) local_acc[0] = local_acc[0] + local_acc[i];
    timer.stop("Parallel Acuumulation");

    assert(local_acc[0] == points[n]);
    printf("Everything is OK!\n");

    return 0;
}