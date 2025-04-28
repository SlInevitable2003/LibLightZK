#pragma once
#include "device-utils.cuh"
#include "xpu-vector.cuh"

template <typename devFdT>
__global__ __launch_bounds__(1024, 1) void light_bsc(
    size_t length, devFdT *scalars, uint32_t *bucket_siz, 
    size_t bucket_cnt, size_t win_siz, size_t win_cnt)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    __shared__ devFdT shmem[1024];
    for (size_t i = tid; i < length; i += stride) {
        shmem[threadIdx.x] = scalars[i];
        shmem[threadIdx.x].from();
        for (size_t j = 0; j < win_cnt; j++) {
            uint32_t bucket_id = extract_bits((const uint32_t*)(shmem + threadIdx.x), 8 * sizeof(devFdT), j * win_siz, (j + 1) * win_siz - 1);
            atomicAdd(bucket_siz + j * bucket_cnt + bucket_id, 1);
        }
    }
}
__global__ void light_bsp(uint32_t *bucket_siz, uint32_t *bucket_top, size_t bucket_cnt)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    bucket_top[tid * bucket_cnt + 0] = 0;
    for (uint32_t i = 1; i < bucket_cnt; i++) {
        bucket_top[tid * bucket_cnt + i] = bucket_top[tid * bucket_cnt + i - 1] + bucket_siz[tid * bucket_cnt + i - 1];
    }
}
template <typename devFdT>
__global__ __launch_bounds__(1024, 1) void light_bscx(
    size_t length, devFdT *scalars, 
    uint32_t *point_idx, uint32_t *bucket_top, 
    size_t bucket_cnt, size_t win_siz, size_t win_cnt)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    __shared__ devFdT shmem[1024];
    for (size_t i = tid; i < length; i += stride) {
        shmem[threadIdx.x] = scalars[i];
        shmem[threadIdx.x].from();
        for (size_t j = 0; j < win_cnt; j++) {
            uint32_t bucket_id = extract_bits((const uint32_t*)(shmem + threadIdx.x), 8 * sizeof(devFdT), j * win_siz, (j + 1) * win_siz - 1);
            uint32_t off = atomicAdd(bucket_top + j * bucket_cnt + bucket_id, 1);
            point_idx[j * length + off] = i;
        }
    }
}
template<typename dG1> 
__global__ __launch_bounds__(512, 1) void light_bag1(
    dG1 *points, uint32_t *point_idx, 
    uint32_t *bucket_siz, uint32_t *bucket_top, dG1 *bucket_sum,
    size_t bucket_cnt, size_t length)
{
    size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) % 32;
    size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    size_t win_id = warp_id / bucket_cnt;
    size_t bucket_tail = bucket_top[warp_id];
    size_t bucket_head = bucket_tail - bucket_siz[warp_id];

    dG1 acc, incr; acc.zero();
    for (size_t i = bucket_head + tid; i < bucket_tail; i += 32) acc.aadd(points[point_idx[win_id * length + i]]);
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        for (uint32_t j = 0; j < sizeof(dG1) / sizeof(uint32_t); j++) 
            ((uint32_t*)&incr)[j] = __shfl_down_sync(0xffffffff, ((uint32_t*)&acc)[j], offset);
        acc.dadd(incr);
    }

    if (tid == 0) bucket_sum[warp_id] = acc;
}
template<typename dG2> 
__global__ __launch_bounds__(512, 1) void light_bag2(
    dG2 *points, uint32_t *point_idx, 
    uint32_t *bucket_siz, uint32_t *bucket_top, dG2 *bucket_sum,
    size_t bucket_cnt, size_t length)
{
    size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) % 32;
    size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    size_t win_id = warp_id / bucket_cnt;
    size_t bucket_tail = bucket_top[warp_id];
    size_t bucket_head = bucket_tail - bucket_siz[warp_id];

    dG2 acc, incr; acc.zero();
    for (size_t i = bucket_head + (tid/2); i < bucket_tail; i += 16) acc.aadd(points[2 * point_idx[win_id * length + i] + (threadIdx.x & 1)]);
    for (size_t offset = 16; offset > 1; offset >>= 1) {
        for (uint32_t j = 0; j < sizeof(dG2) / sizeof(uint32_t); j++) 
            ((uint32_t*)&incr)[j] = __shfl_down_sync(0xffffffff, ((uint32_t*)&acc)[j], offset);
        acc.dadd(incr);
    }

    if (tid <= 1) bucket_sum[2 * warp_id + (threadIdx.x & 1)] = acc;
}
template<typename dG1> 
__global__ __launch_bounds__(1024, 1) void light_bsg1(dG1 *buckets_sum, size_t bucket_cnt, size_t win_cnt)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < bucket_cnt * win_cnt; i += stride) {
        dG1 acc, incr = buckets_sum[i]; acc.zero();
        for (size_t j = bucket_cnt / 2; j > 0; j >>= 1) {
            acc.dbl();
            if ((i % bucket_cnt) & j) acc.dadd(incr);
        }
        buckets_sum[i] = acc;
    }
}
template<typename dG2> 
__global__ __launch_bounds__(1024, 1) void light_bsg2(dG2 *buckets_sum, size_t bucket_cnt, size_t win_cnt)
{
    size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) / 2, stride = (blockDim.x * gridDim.x) / 2;
    for (size_t i = tid; i < bucket_cnt * win_cnt; i += stride) {
        dG2 acc, incr = buckets_sum[2 * i + (threadIdx.x & 1)]; acc.zero();
        for (size_t j = bucket_cnt / 2; j > 0; j >>= 1) {
            acc.dbl();
            if ((i % bucket_cnt) & j) acc.dadd(incr);
        }
        buckets_sum[2 * i + (threadIdx.x & 1)] = acc;
    }
}
template<typename dG1> 
__global__ __launch_bounds__(1024, 1) void light_brg1(dG1 *buckets_sum, uint32_t bucket_cnt, dG1 *win_sum)
{
    size_t tid = threadIdx.x % 32, wid = threadIdx.x / 32;

    dG1 acc, incr; acc.zero();
    for (size_t i = tid; i < bucket_cnt; i += 32) acc.dadd(buckets_sum[wid * bucket_cnt + i]);
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        for (size_t j = 0; j < sizeof(dG1) / sizeof(uint32_t); j++) 
            ((uint32_t*)&incr)[j] = __shfl_down_sync(0xffffffff, ((uint32_t*)&acc)[j], offset);
        acc.dadd(incr);
    }

    if (tid == 0) win_sum[wid] = acc;
}
template<typename dG2> 
__global__ __launch_bounds__(1024, 1) void light_brg2(dG2 *buckets_sum, uint32_t bucket_cnt, dG2 *win_sum)
{
    size_t tid = threadIdx.x % 32, wid = threadIdx.x / 32;

    dG2 acc, incr; acc.zero();
    for (size_t i = tid; i < 2 * bucket_cnt; i += 32) acc.dadd(buckets_sum[wid * 2 * bucket_cnt + i]);
    for (size_t offset = 16; offset > 1; offset >>= 1) {
        for (size_t j = 0; j < sizeof(dG2) / sizeof(uint32_t); j++) 
            ((uint32_t*)&incr)[j] = __shfl_down_sync(0xffffffff, ((uint32_t*)&acc)[j], offset);
        acc.dadd(incr);
    }

    if (tid <= 1) win_sum[wid * 2 + (threadIdx.x & 1)] = acc;
}
template<typename dG1> 
__global__ void light_wrg1(dG1 *win_sum, size_t win_siz, size_t win_cnt, dG1* dest)
{
    dG1 acc; acc.zero();
    for (size_t i = win_cnt; i > 0; i--) {
        for (size_t j = 0; j < win_siz; j++) acc.dbl();
        acc.dadd(win_sum[i - 1]);
    }
    *dest = acc;
}
template<typename dG2> 
__global__ void light_wrg2(dG2 *win_sum, size_t win_siz, size_t win_cnt, dG2* dest)
{
    dG2 acc; acc.zero();
    for (size_t i = win_cnt; i > 0; i--) {
        for (size_t j = 0; j < win_siz; j++) acc.dbl();
        acc.dadd(win_sum[2 * (i - 1) + (threadIdx.x & 1)]);
    }
    dest[threadIdx.x] = acc;
}
template<typename devFdT, typename dG1>
void light_msmg1(
    size_t length, devFdT *scalars, dG1 *points, dG1 *dest, 
    size_t win_siz, size_t win_cnt, size_t bucket_cnt)
{
    xpu::vector<uint32_t> bucket_siz(win_cnt * bucket_cnt, xpu::mem_policy::cross_platform),
                          bucket_top(win_cnt * bucket_cnt, xpu::mem_policy::device_only),
                          point_idx(win_cnt * length, xpu::mem_policy::device_only);
    xpu::vector<dG1> bucket_sum(win_cnt * bucket_cnt, xpu::mem_policy::device_only),
                     win_sum(win_cnt, xpu::mem_policy::device_only);

    bucket_siz.clear();
    (light_bsc<devFdT>)<<<14, 1024>>>(length, scalars, (uint32_t*)bucket_siz.p(), bucket_cnt, win_siz, win_cnt);
    light_bsp<<<1, win_cnt>>>((uint32_t*)bucket_siz.p(), (uint32_t*)bucket_top.p(), bucket_cnt);
    (light_bscx<devFdT>)<<<14, 1024>>>(length, scalars, (uint32_t*)point_idx.p(), (uint32_t*)bucket_top.p(), bucket_cnt, win_siz, win_cnt);
    (light_bag1<dG1>)<<<512, 512>>>(points, (uint32_t*)point_idx.p(), (uint32_t*)bucket_siz.p(), (uint32_t*)bucket_top.p(), (dG1*)bucket_sum.p(), bucket_cnt, length);
    (light_bsg1<dG1>)<<<1, 1024>>>((dG1*)bucket_sum.p(), bucket_cnt, win_cnt);
    (light_brg1<dG1>)<<<1, 1024>>>((dG1*)bucket_sum.p(), bucket_cnt, (dG1*)win_sum.p());
    (light_wrg1<dG1>)<<<1, 1>>>((dG1*)win_sum.p(), win_siz, win_cnt, dest);
}
template<typename devFdT, typename dG2>
void light_msmg2(
    size_t length, devFdT *scalars, dG2 *points, dG2 *dest, 
    size_t win_siz, size_t win_cnt, size_t bucket_cnt)
{
    xpu::vector<uint32_t> bucket_siz(win_cnt * bucket_cnt, xpu::mem_policy::cross_platform),
                          bucket_top(win_cnt * bucket_cnt, xpu::mem_policy::device_only),
                          point_idx(win_cnt * length, xpu::mem_policy::device_only);
    xpu::vector<dG2> bucket_sum(2 * win_cnt * bucket_cnt, xpu::mem_policy::device_only),
                     win_sum(2 * win_cnt, xpu::mem_policy::device_only);

    bucket_siz.clear();
    (light_bsc<devFdT>)<<<216, 1024>>>(length, scalars, (uint32_t*)bucket_siz.p(), bucket_cnt, win_siz, win_cnt);
    light_bsp<<<1, win_cnt>>>((uint32_t*)bucket_siz.p(), (uint32_t*)bucket_top.p(), bucket_cnt);
    (light_bscx<devFdT>)<<<216, 1024>>>(length, scalars, (uint32_t*)point_idx.p(), (uint32_t*)bucket_top.p(), bucket_cnt, win_siz, win_cnt);
    (light_bag2<dG2>)<<<512, 512>>>(points, (uint32_t*)point_idx.p(), (uint32_t*)bucket_siz.p(), (uint32_t*)bucket_top.p(), (dG2*)bucket_sum.p(), bucket_cnt, length);
    (light_bsg2<dG2>)<<<1, 1024>>>((dG2*)bucket_sum.p(), bucket_cnt, win_cnt);
    (light_brg2<dG2>)<<<1, 1024>>>((dG2*)bucket_sum.p(), bucket_cnt, (dG2*)win_sum.p());
    (light_wrg2<dG2>)<<<1, 2>>>((dG2*)win_sum.p(), win_siz, win_cnt, dest);
}