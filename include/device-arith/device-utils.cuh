#pragma once
#include <iostream>
#define CUDA_DEBUG cudaDeviceSynchronize(); std::cout << __LINE__ << " " << cudaGetErrorString(cudaGetLastError()) << std::endl;

__device__ __host__ inline void vec_select(void* dst, const void* a, const void* b, size_t size, bool cond) 
{
    char* d = static_cast<char*>(dst);
    const char* src = cond ? static_cast<const char*>(a) : static_cast<const char*>(b);
    for (size_t i = 0; i < size; i++) d[i] = src[i];
}
__device__ uint32_t extract_bits(const uint32_t* big_int, const size_t total_bits, size_t lo, size_t hi) 
{
    if (hi >= total_bits) hi = total_bits - 1;
    const int bit_width = hi - lo + 1;

    const int start_elem = lo / 32;
    const int start_bit = lo % 32;

    uint32_t low_word = big_int[start_elem];
    uint32_t high_word = ((start_elem + 1) < (total_bits / 32)) ? big_int[start_elem + 1] : 0;
    uint64_t combined = ((uint64_t)high_word << 32) | low_word;

    uint64_t mask = (1ULL << bit_width) - 1;
    return (combined >> start_bit) & mask;
}

__device__ void lock(int *mutex) {
    while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0);
}