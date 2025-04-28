#include "device-utils.cuh"
#include "timer.cuh"
#include "groth16-operator.cuh"
#include "alt_bn128_t.cuh"
#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"
#include "omp.h"

const size_t n = 1 << 20;
const size_t win_siz = 8;
const size_t win_cnt = (256 + win_siz - 1) / win_siz;
const size_t bucket_cnt = 1 << win_siz;

__global__ void layout_switch(alt_bn128::g2_t *dest, alt_bn128::g2_t *src, size_t length)
{
    size_t tid = threadIdx.x + blockIdx.x + blockDim.x, stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < 2*length; i += stride) src[i].write_to(dest + (i&(~1)));
}

int main(int argc, char *argv[])
{
    Timer timer;

    libff::init_alt_bn128_params();

    xpu::vector<libff::alt_bn128_G2> dev_points(n, xpu::mem_policy::device_only), dev_res(1, xpu::mem_policy::cross_platform);
    xpu::vector<libff::alt_bn128_Fr> dev_scalars(n, xpu::mem_policy::cross_platform);

    xpu::vector<libff::alt_bn128_G2> dev_uni(1, xpu::mem_policy::cross_platform);
    dev_uni[0] = libff::alt_bn128_G2::one();
    dev_uni.store();

    timer.start();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_scalars[i] = libff::alt_bn128_Fr::random_element();
    dev_scalars.store();

    (fix_base_multi_scalar_multiplication_g2<alt_bn128::fr_t, alt_bn128::g2_t>)<<<14, 510>>>((alt_bn128::g2_t*)dev_points.p(), (alt_bn128::fr_t*)dev_scalars.p(), n, (alt_bn128::g2_t*)dev_uni.p());
    CUDA_DEBUG;
    timer.stop();

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_scalars[i] = libff::alt_bn128_Fr::random_element();
    dev_scalars.store();

    CUDA_DEBUG;
    timer.start();
    light_msmg2<alt_bn128::fr_t, alt_bn128::g2_t>(n, (alt_bn128::fr_t*)dev_scalars.p(), (alt_bn128::g2_t*)dev_points.p(), (alt_bn128::g2_t*)dev_res.p(), win_siz, win_cnt, bucket_cnt);
    CUDA_DEBUG;
    timer.stop();
    dev_res.load();

    xpu::vector<libff::alt_bn128_G2> points(n, xpu::mem_policy::cross_platform);
    layout_switch<<<14, 1024>>>((alt_bn128::g2_t*)points.p(), (alt_bn128::g2_t*)dev_points.p(), n);
    const int num_threads = omp_get_max_threads();
    timer.start();
    std::vector<libff::alt_bn128_G2> local_acc(num_threads, libff::alt_bn128_G2::zero());
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n; i++) local_acc[tid] = local_acc[tid] + dev_scalars[i] * points[i];
    }
    for (size_t i = 1; i < num_threads; i++) local_acc[0] = local_acc[0] + local_acc[i];
    timer.stop("Parallel Acuumulation");

    assert(local_acc[0] == dev_res[0]);
    printf("Everything is OK!\n");

    return 0;
}