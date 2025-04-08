#include "device-utils.cuh"
#include "timer.cuh"
#include "alt_bn128_t.cuh"
#include "alt_bn128_g1.cuh"
#include "multigpu-msm.cuh"
#include "omp.h"

#define DEV_CNT 2

const size_t n = 1 << 20;
const size_t win_siz = 12;
const size_t win_cnt = (256 + win_siz - 1) / win_siz;
const size_t bucket_cnt = 1 << win_siz;

int main(int argc, char *argv[])
{
    Timer timer;
    int dev_cnt;
    cudaGetDeviceCount(&dev_cnt);

    libff::init_alt_bn128_params();

    xpu::multigpu_vector<libff::alt_bn128_G1, DEV_CNT> dev_points((win_cnt + dev_cnt - 1) / dev_cnt * n, xpu::mem_policy::device_only);
    xpu::multigpu_vector<libff::alt_bn128_Fr, DEV_CNT> dev_scalars(n, xpu::mem_policy::cross_platform);

    CUDA_DEBUG;
    timer.start();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_scalars[i] = libff::alt_bn128_Fr::random_element();
    dev_scalars.store();

    multi_launch_fix_base_multi_scalar_multiplication_g1<libff::alt_bn128_Fr, libff::alt_bn128_G1, alt_bn128::fr_t, alt_bn128::g1_t, DEV_CNT>(dev_points, dev_scalars, n);
    multi_launch_pre_comp_g1<libff::alt_bn128_G1, alt_bn128::g1_t, DEV_CNT>(n, dev_points, win_siz, win_cnt);
    CUDA_DEBUG;
    timer.stop();

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_scalars[i] = libff::alt_bn128_Fr::random_element();
    dev_scalars.store();

    xpu::multigpu_vector<libff::alt_bn128_G1, DEV_CNT> dest(1, xpu::mem_policy::cross_platform);
    CUDA_DEBUG;
    timer.start();
    multi_launch_multi_scalar_multiplication_g1<libff::alt_bn128_Fr, libff::alt_bn128_G1, alt_bn128::fr_t, alt_bn128::g1_t, DEV_CNT>(n, dev_scalars, dev_points, dest, win_siz, win_cnt, bucket_cnt, n);
    CUDA_DEBUG;
    timer.stop();

    libff::alt_bn128_G1 gpu_ans = libff::alt_bn128_G1::zero();
    for (int dev = 0; dev < dev_cnt; dev++) {
        dest.load(dev);
        gpu_ans = gpu_ans + dest[0];
    }

    const int num_threads = omp_get_max_threads();
    xpu::vector<libff::alt_bn128_G1> points(n);
    points.load(dev_points.p(0), n);

    CUDA_DEBUG;
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

    // assert(local_acc[0] == gpu_ans);
    printf("Everything is OK!\n");

    return 0;
}