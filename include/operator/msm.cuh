#pragma once
#include "device-utils.cuh"
#include "xpu-vector.cuh"

template <typename devFdT, typename dG1>
__global__ __launch_bounds__(510, 3) void fix_base_multi_scalar_multiplication_g1(dG1 *dest, devFdT *scalars, const size_t length)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = gridDim.x * blockDim.x;

    __shared__ dG1 shmem[510];
    __shared__ dG1 uni;
    if (threadIdx.x == 0) uni.one();
    __syncthreads();

    for (size_t i = tid; i < length; i += stride) {
        shmem[threadIdx.x].zero(); scalars[i].from();
        uint32_t *p = (uint32_t *)(scalars + i);
        for (size_t j = 8 * sizeof(devFdT); j > 0; j--) {
            shmem[threadIdx.x].dbl();
            if ((p[(j - 1) / 32] >> ((j - 1) % 32)) & 1) shmem[threadIdx.x].dadd(uni);
        }
        dest[i] = shmem[threadIdx.x];
    }
}

template <typename devFdT, typename dG2>
__global__ __launch_bounds__(510, 3) void fix_base_multi_scalar_multiplication_g2(dG2 *dest, devFdT *scalars, const size_t length, dG2 *uni_raw)
{
    size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) / 2, stride = gridDim.x * blockDim.x / 2;

    __shared__ dG2 shmem[510];
    __shared__ dG2 uni;
    if (threadIdx.x == 0) uni.read_from(uni_raw);
    __syncthreads();

    for (size_t i = tid; i < length; i += stride) {
        shmem[threadIdx.x].zero(); 
        if (threadIdx.x & 1) scalars[i].from();
        uint32_t *p = (uint32_t *)(scalars + i);
        for (size_t j = 8 * sizeof(devFdT); j > 0; j--) {
            shmem[threadIdx.x].dbl();
            if ((p[(j - 1) / 32] >> ((j - 1) % 32)) & 1) shmem[threadIdx.x].dadd(dest[2 * i + (threadIdx.x & 1)]);
        }
        dest[2 * i + (threadIdx.x & 1)] = shmem[threadIdx.x];
    }
}

template<typename dG1> 
__global__ __launch_bounds__(1024, 2) void pre_comp_g1(size_t length, dG1 *points, size_t gap, size_t level)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < length; i += stride) {
        points[i].to_affine();
        dG1 cur = points[i];
        for (size_t j = 1; j < level; j++) {
            for (size_t k = 0; k < gap; k++) cur.dbl();
            points[i + j * length] = cur;
            points[i + j * length].to_affine();
        }
    }
}
template<typename dG2> 
__global__ __launch_bounds__(1024, 2) void pre_comp_g2(size_t length, dG2 *points, size_t gap, size_t level)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < 2 * length; i += stride) {
        points[i].to_affine();
        dG2 cur = points[i];
        for (size_t j = 1; j < level; j++) {
            for (size_t k = 0; k < gap; k++) cur.dbl();
            points[i + 2 * j * length] = cur;
            points[i + 2 * j * length].to_affine();
        }
    }
}

template <typename devFdT>
__global__ __launch_bounds__(1024, 2) void bucket_scatter_count(size_t length, devFdT *scalars, uint32_t *bucket_siz, size_t win_siz, size_t level)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    __shared__ devFdT shmem[1024];
    for (size_t i = tid; i < length; i += stride) {
        shmem[threadIdx.x] = scalars[i];
        shmem[threadIdx.x].from();
        for (size_t j = 0; j < level; j++) {
            uint32_t bucket_id = extract_bits((const uint32_t*)(shmem + threadIdx.x), 8 * sizeof(devFdT), j * win_siz, (j + 1) * win_siz - 1);
            atomicAdd(bucket_siz + bucket_id, 1);
        }
    }
}
__global__ void bucket_scatter_prefixsum(uint32_t *bucket_siz, uint32_t *bucket_top, size_t bucket_cnt)
{
    bucket_top[0] = 0;
    for (uint32_t i = 1; i < bucket_cnt; i++) {
        bucket_top[i] = bucket_top[i - 1] + bucket_siz[i - 1];
    }
}
template <typename devFdT>
__global__ __launch_bounds__(1024, 2) void bucket_scatter_calidx(
    size_t length, devFdT *scalars, 
    uint32_t *point_idx, uint32_t *bucket_top, 
    size_t win_siz, size_t level, size_t level_stride)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    __shared__ devFdT shmem[1024];
    for (size_t i = tid; i < length; i += stride) {
        shmem[threadIdx.x] = scalars[i];
        shmem[threadIdx.x].from();
        for (size_t j = 0; j < level; j++) {
            uint32_t bucket_id = extract_bits((const uint32_t*)(shmem + threadIdx.x), 8 * sizeof(devFdT), j * win_siz, (j + 1) * win_siz - 1);
            uint32_t off = atomicAdd(bucket_top + bucket_id, 1);
            point_idx[off] = i + j * level_stride;
        }
    }
}
template<typename dG1> 
__global__ __launch_bounds__(1024, 2) void bucket_accumulation_g1(
    dG1 *points, uint32_t *point_idx, 
    uint32_t *bucket_siz, uint32_t *bucket_top, dG1 *bucket_sum)
{
    size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) % 32, bucket_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    size_t bucket_tail = bucket_top[bucket_id];

    dG1 acc, incr; acc.zero();
    for (size_t i = bucket_tail - bucket_siz[bucket_id] + tid; i < bucket_tail; i += 32) acc.aadd(points[point_idx[i]]);
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        for (uint32_t j = 0; j < sizeof(dG1) / sizeof(uint32_t); j++) 
            ((uint32_t*)&incr)[j] = __shfl_down_sync(0xffffffff, ((uint32_t*)&acc)[j], offset);
        acc.dadd(incr);
    }

    if (tid == 0) bucket_sum[bucket_id] = acc;
}
template<typename dG2> 
__global__ __launch_bounds__(1024, 2) void bucket_accumulation_g2(
    dG2 *points, uint32_t *point_idx, 
    uint32_t *bucket_siz, uint32_t *bucket_top, dG2 *bucket_sum)
{
    size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) % 32, bucket_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    size_t bucket_tail = bucket_top[bucket_id];

    dG2 acc, incr; acc.zero();
    for (size_t i = bucket_tail - bucket_siz[bucket_id] + (tid/2); i < bucket_tail; i += 16) acc.aadd(points[2 * point_idx[i] + (threadIdx.x & 1)]);
    for (size_t offset = 16; offset > 1; offset >>= 1) {
        for (uint32_t j = 0; j < sizeof(dG2) / sizeof(uint32_t); j++) 
            ((uint32_t*)&incr)[j] = __shfl_down_sync(0xffffffff, ((uint32_t*)&acc)[j], offset);
        acc.dadd(incr);
    }

    if (tid <= 1) bucket_sum[2 * bucket_id + (threadIdx.x & 1)] = acc;
}
template<typename dG1> 
__global__ __launch_bounds__(64, 1) void bucket_scale_g1(dG1 *buckets_sum, size_t bucket_cnt)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < bucket_cnt; i += stride) {
        dG1 acc, incr = buckets_sum[i]; acc.zero();
        for (size_t j = bucket_cnt / 2; j > 0; j >>= 1) {
            acc.dbl();
            if (i & j) acc.dadd(incr);
        }
        buckets_sum[i] = acc;
    }
}
template<typename dG2> 
__global__ __launch_bounds__(64, 1) void bucket_scale_g2(dG2 *buckets_sum, size_t bucket_cnt)
{
    size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) / 2, stride = (blockDim.x * gridDim.x) / 2;
    for (size_t i = tid; i < bucket_cnt; i += stride) {
        dG2 acc, incr = buckets_sum[2 * i + (threadIdx.x & 1)]; acc.zero();
        for (size_t j = bucket_cnt / 2; j > 0; j >>= 1) {
            acc.dbl();
            if (i & j) acc.dadd(incr);
        }
        buckets_sum[2 * i + (threadIdx.x & 1)] = acc;
    }
}

template<typename dG1> 
__global__ __launch_bounds__(32, 1) void bucket_reduce_g1(dG1 *buckets_sum, uint32_t bucket_cnt, dG1 *dest)
{
    size_t tid = threadIdx.x;

    dG1 acc, incr; acc.zero();
    for (size_t i = tid; i < bucket_cnt; i += 32) acc.dadd(buckets_sum[i]);
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        for (size_t j = 0; j < sizeof(dG1) / sizeof(uint32_t); j++) 
            ((uint32_t*)&incr)[j] = __shfl_down_sync(0xffffffff, ((uint32_t*)&acc)[j], offset);
        acc.dadd(incr);
    }

    if (tid == 0) *dest = acc;
}
template<typename dG2> 
__global__ __launch_bounds__(32, 1) void bucket_reduce_g2(dG2 *buckets_sum, uint32_t bucket_cnt, dG2 *dest)
{
    size_t tid = threadIdx.x;

    dG2 acc, incr; acc.zero();
    for (size_t i = tid; i < 2 * bucket_cnt; i += 32) acc.dadd(buckets_sum[i]);
    for (size_t offset = 16; offset > 1; offset >>= 1) {
        for (size_t j = 0; j < sizeof(dG2) / sizeof(uint32_t); j++) 
            ((uint32_t*)&incr)[j] = __shfl_down_sync(0xffffffff, ((uint32_t*)&acc)[j], offset);
        acc.dadd(incr);
    }

    if (tid <= 1) acc.write_to((void*)dest);
}

template<typename devFdT, typename dG1>
void multi_scalar_multiplication_g1(
    size_t length, devFdT *scalars, dG1 *points, dG1 *dest, 
    size_t win_siz, size_t win_cnt, size_t bucket_cnt, 
    size_t level_stride)
{
    xpu::vector<uint32_t> bucket_siz(bucket_cnt, xpu::mem_policy::device_only),
                          bucket_top(bucket_cnt, xpu::mem_policy::device_only),
                          point_idx(length * win_cnt, xpu::mem_policy::device_only);
    xpu::vector<dG1> bucket_sum(bucket_cnt, xpu::mem_policy::device_only);

    bucket_siz.clear();
    (bucket_scatter_count<devFdT>)<<<216, 1024>>>(length, scalars, (uint32_t*)bucket_siz.p(), win_siz, win_cnt);
    bucket_scatter_prefixsum<<<1, 1>>>((uint32_t*)bucket_siz.p(), (uint32_t*)bucket_top.p(), bucket_cnt);
    (bucket_scatter_calidx<devFdT>)<<<216, 1024>>>(length, scalars, (uint32_t*)point_idx.p(), (uint32_t*)bucket_top.p(), win_siz, win_cnt, level_stride);
    (bucket_accumulation_g1<dG1>)<<<128, 1024>>>(points, (uint32_t*)point_idx.p(), (uint32_t*)bucket_siz.p(), (uint32_t*)bucket_top.p(), (dG1*)bucket_sum.p());
    (bucket_scale_g1<dG1>)<<<64, 64>>>((dG1*)bucket_sum.p(), bucket_cnt);
    (bucket_reduce_g1<dG1>)<<<1, 32>>>((dG1*)bucket_sum.p(), bucket_cnt, dest);
}

template<typename devFdT, typename dG2>
void multi_scalar_multiplication_g2(
    size_t length, devFdT *scalars, dG2 *points, dG2 *dest, 
    size_t win_siz, size_t win_cnt, size_t bucket_cnt, 
    size_t level_stride)
{
    xpu::vector<uint32_t> bucket_siz(bucket_cnt, xpu::mem_policy::device_only),
                          bucket_top(bucket_cnt, xpu::mem_policy::device_only),
                          point_idx(length * win_cnt, xpu::mem_policy::device_only);
    xpu::vector<dG2> bucket_sum(2 * bucket_cnt, xpu::mem_policy::device_only);

    bucket_siz.clear();
    (bucket_scatter_count<devFdT>)<<<216, 1024>>>(length, scalars, (uint32_t*)bucket_siz.p(), win_siz, win_cnt);
    bucket_scatter_prefixsum<<<1, 1>>>((uint32_t*)bucket_siz.p(), (uint32_t*)bucket_top.p(), bucket_cnt);
    (bucket_scatter_calidx<devFdT>)<<<216, 1024>>>(length, scalars, (uint32_t*)point_idx.p(), (uint32_t*)bucket_top.p(), win_siz, win_cnt, level_stride);
    (bucket_accumulation_g2<dG2>)<<<128, 1024>>>(points, (uint32_t*)point_idx.p(), (uint32_t*)bucket_siz.p(), (uint32_t*)bucket_top.p(), (dG2*)bucket_sum.p());
    (bucket_scale_g2<dG2>)<<<64, 64>>>((dG2*)bucket_sum.p(), bucket_cnt);
    (bucket_reduce_g2<dG2>)<<<1, 32>>>((dG2*)bucket_sum.p(), bucket_cnt, dest);
}