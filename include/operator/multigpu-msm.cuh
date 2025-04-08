#pragma once
#include "msm.cuh"
#include <vector>
#include "xpu-multigpu-vector.cuh"

template <typename FieldT, typename G1T, typename devFdT, typename dG1, int dev_cnt>
void multi_launch_fix_base_multi_scalar_multiplication_g1(xpu::multigpu_vector<G1T, dev_cnt> &dest, xpu::multigpu_vector<FieldT, dev_cnt> &scalars, const size_t length)
{
    #pragma omp parallel for
    for (int dev = 0; dev < dev_cnt; dev++) {
        cudaSetDevice(dev);
        (fix_base_multi_scalar_multiplication_g1<devFdT, dG1>)<<<324, 510>>>((dG1*)dest.p(dev), (devFdT*)scalars.p(dev), length);
        cudaDeviceSynchronize();
    }
}

template <typename FieldT, typename G2T, typename devFdT, typename dG2, int dev_cnt>
void multi_launch_fix_base_multi_scalar_multiplication_g2(xpu::multigpu_vector<G2T, dev_cnt> &dest, xpu::multigpu_vector<FieldT, dev_cnt> &scalars, const size_t length)
{
    #pragma omp parallel for
    for (int dev = 0; dev < dev_cnt; dev++) {
        cudaSetDevice(dev);
        (fix_base_multi_scalar_multiplication_g2<devFdT, dG2>)<<<324, 510>>>((dG2*)dest.p(dev), (devFdT*)scalars.p(dev), length);
        cudaDeviceSynchronize();
    }
}

template<typename dG1> 
__global__ __launch_bounds__(1024, 2) void pre_comp_g1_part(size_t length, dG1 *points, size_t gap, size_t level_lo, size_t level_hi)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < length; i += stride) {
        points[i].to_affine();
        dG1 cur = points[i];
        for (size_t j = 1; j < level_lo; j++) for (size_t k = 0; k < gap; k++) cur.dbl();
        for (size_t j = level_lo + (level_lo == 0); j < level_hi; j++) {
            for (size_t k = 0; k < gap; k++) cur.dbl();
            points[i + (j - level_lo) * length] = cur;
            points[i + (j - level_lo) * length].to_affine();
        }
    }
}
template <typename G1T, typename dG1, int dev_cnt>
void multi_launch_pre_comp_g1(size_t length, xpu::multigpu_vector<G1T, dev_cnt> &points, size_t gap, size_t level)
{
    size_t dev_gap = (level + dev_cnt - 1) / dev_cnt;
    #pragma omp parallel for
    for (int dev = 0; dev < dev_cnt; dev++) {
        cudaSetDevice(dev);
        (pre_comp_g1_part<dG1>)<<<216, 1024>>>(length, (dG1*)points.p(dev), gap, dev * dev_gap, min((dev + 1) * dev_gap, level));
        cudaDeviceSynchronize();
    }
}

template <typename devFdT>
__global__ __launch_bounds__(1024, 2) void bucket_scatter_count_part(size_t length, devFdT *scalars, uint32_t *bucket_siz, size_t win_siz, size_t level_lo, size_t level_hi)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    __shared__ devFdT shmem[1024];
    for (size_t i = tid; i < length; i += stride) {
        shmem[threadIdx.x] = scalars[i];
        shmem[threadIdx.x].from();
        for (size_t j = level_lo; j < level_hi; j++) {
            uint32_t bucket_id = extract_bits((const uint32_t*)(shmem + threadIdx.x), 8 * sizeof(devFdT), j * win_siz, (j + 1) * win_siz - 1);
            atomicAdd(bucket_siz + bucket_id, 1);
        }
    }
}
template <typename devFdT>
__global__ __launch_bounds__(1024, 2) void bucket_scatter_calidx_part(
    size_t length, devFdT *scalars, 
    uint32_t *point_idx, uint32_t *bucket_top, 
    size_t win_siz, size_t level_lo, size_t level_hi, size_t level_stride)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    __shared__ devFdT shmem[1024];
    for (size_t i = tid; i < length; i += stride) {
        shmem[threadIdx.x] = scalars[i];
        shmem[threadIdx.x].from();
        for (size_t j = level_lo; j < level_hi; j++) {
            uint32_t bucket_id = extract_bits((const uint32_t*)(shmem + threadIdx.x), 8 * sizeof(devFdT), j * win_siz, (j + 1) * win_siz - 1);
            uint32_t off = atomicAdd(bucket_top + bucket_id, 1);
            point_idx[off] = i + (j - level_lo) * level_stride;
        }
    }
}
template <typename FieldT, typename G1T, typename devFdT, typename dG1, int dev_cnt>
void multi_launch_multi_scalar_multiplication_g1(
    size_t length, 
    xpu::multigpu_vector<FieldT, dev_cnt> &scalars, 
    xpu::multigpu_vector<G1T, dev_cnt> &points, 
    xpu::multigpu_vector<G1T, dev_cnt> &dest, 
    size_t win_siz, size_t win_cnt, size_t bucket_cnt, 
    size_t level_stride)
{
    xpu::multigpu_vector<uint32_t, dev_cnt> bucket_siz(bucket_cnt, xpu::mem_policy::device_only),
                                            bucket_top(bucket_cnt, xpu::mem_policy::device_only),
                                            point_idx(length * (win_cnt + dev_cnt - 1) / dev_cnt, xpu::mem_policy::device_only);
    xpu::multigpu_vector<dG1, dev_cnt> bucket_sum(bucket_cnt, xpu::mem_policy::device_only);

    bucket_siz.clear();
    size_t dev_gap = (win_cnt + dev_cnt - 1) / dev_cnt;
    #pragma omp parallel for
    for (int dev = 0; dev < dev_cnt; dev++) {
        cudaSetDevice(dev);
        (bucket_scatter_count_part<devFdT>)<<<160, 1024>>>(length, (devFdT*)scalars.p(dev), (uint32_t*)bucket_siz.p(dev), win_siz, dev * dev_gap, min((dev + 1) * dev_gap, win_cnt));
        bucket_scatter_prefixsum<<<1, 1>>>((uint32_t*)bucket_siz.p(dev), (uint32_t*)bucket_top.p(dev), bucket_cnt);
        (bucket_scatter_calidx_part<devFdT>)<<<160, 1024>>>(length, (devFdT*)scalars.p(dev), (uint32_t*)point_idx.p(dev), (uint32_t*)bucket_top.p(dev), win_siz, dev * dev_gap, min((dev + 1) * dev_gap, win_cnt), level_stride);
        (bucket_accumulation_g1<dG1>)<<<128, 1024>>>((dG1*)points.p(dev), (uint32_t*)point_idx.p(dev), (uint32_t*)bucket_siz.p(dev), (uint32_t*)bucket_top.p(dev), (dG1*)bucket_sum.p(dev));
        (bucket_scale_g1<dG1>)<<<64, 64>>>((dG1*)bucket_sum.p(dev), bucket_cnt);
        (bucket_reduce_g1<dG1>)<<<1, 32>>>((dG1*)bucket_sum.p(dev), bucket_cnt, (dG1*)dest.p(dev));
        cudaDeviceSynchronize();
    }
}
