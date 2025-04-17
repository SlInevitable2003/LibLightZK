#include "device-utils.cuh"
#include "timer.cuh"
#include "groth16-operator.cuh"
#include "alt_bn128_t.cuh"
#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"
#include "omp.h"

#include "alt_bn128_init.cuh"

const size_t n = 1 << 18;
const size_t win_siz = log2(n + n / 2) - 9;
const size_t win_cnt = (256 + win_siz - 1) / win_siz;
const size_t bucket_cnt = 1 << win_siz;

int main(int argc, char *argv[])
{
    Timer timer;

    libff::init_alt_bn128_params();
    timer.start();
    xpu::vector<libff::alt_bn128_G2> dev_points(win_cnt * n, xpu::mem_policy::cross_platform);  //device变量
    xpu::vector<libff::alt_bn128_G2> check_points(n*2);
    xpu::vector<libff::alt_bn128_Fr> dev_scalars(n, xpu::mem_policy::cross_platform);    //混合变量
    CUDA_DEBUG;
    timer.stop("allocate");

    timer.start();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_scalars[i] = libff::alt_bn128_Fr::random_element();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_points[i] = libff::alt_bn128_G2::G2_one;
   
    dev_scalars.store();    
    dev_points.store();
    

    (xxyy_to_xy<alt_bn128::g2_t>)<<<1,512>>>((alt_bn128::g2_t*)dev_points.p(),n);
    
    (fix_base_multi_scalar_multiplication_g2<alt_bn128::fr_t, alt_bn128::g2_t>)<<<160, 510>>>((alt_bn128::g2_t*)dev_points.p(), (alt_bn128::fr_t*)dev_scalars.p(), n);
    
    CUDA_DEBUG;
    timer.stop("Fix Base Multi-Scalar Multiplication");


    // (xy_to_xxyy<alt_bn128::g2_t>)<<<1,512>>>((alt_bn128::g2_t*)dev_points.p(),n);
    // dev_points.load();

    // for(int i=0;i<n;i++){
    //     //printf("%d\n",i);
    //     check_points[i] = libff::alt_bn128_G2::G2_one;
    //     check_points[i] = dev_scalars[i] * check_points[i];
    //     assert(check_points[i] == dev_points[i]);
    // }

    // (xy_to_xxyy<alt_bn128::g2_t>)<<<1,512>>>((alt_bn128::g2_t*)dev_points.p(),n);
    // check_points.load(dev_points.p(), n);
    // (xxyy_to_xy<alt_bn128::g2_t>)<<<1,512>>>((alt_bn128::g2_t*)dev_points.p(),n);
    
    timer.start();
    (pre_comp_g2<alt_bn128::g2_t>)<<<160, 1024>>>(n, (alt_bn128::g2_t*)dev_points.p(), win_siz, win_cnt);
    CUDA_DEBUG;
    timer.stop("Pre-computation");

    // (xy_to_xxyy<alt_bn128::g2_t>)<<<1,512>>>((alt_bn128::g2_t*)dev_points.p(),n*2);
    // dev_points.load();

    // for(int i=0;i<n;i++){
    //     check_points[i+n] = check_points[i];
    //     for(int j=0;j<win_siz;j++){
    //         check_points[i+n] = check_points[i+n].dbl();
    //     }
    //     assert(check_points[i+n] == dev_points[i+n]);
    //     printf("ok\n");
    // }


    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) dev_scalars[i] = libff::alt_bn128_Fr::random_element();
    dev_scalars.store();

    CUDA_DEBUG;
    timer.start();
    multi_scalar_multiplication_g2<alt_bn128::fr_t, alt_bn128::g2_t>(n, (alt_bn128::fr_t*)dev_scalars.p(), (alt_bn128::g2_t*)dev_points.p(), (alt_bn128::g2_t*)dev_points.p() + 2*n, win_siz, win_cnt, bucket_cnt, n);
    CUDA_DEBUG;
    timer.stop("pippenger");

    //dev_points.store(n);
    (xy_to_xxyy<alt_bn128::g2_t>)<<<1,512>>>((alt_bn128::g2_t*)dev_points.p(),n);
    
    xpu::vector<libff::alt_bn128_G2> points(n+1);
    points.load(dev_points.p(), n+1);
    
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

    assert(local_acc[0] == points[n]);
    printf("Everything is OK!\n");

    return 0;

}
