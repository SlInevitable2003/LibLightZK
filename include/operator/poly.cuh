#pragma once
#include "xpu-spmat.cuh"
#include "device-utils.cuh"

template <typename devFdT>
__global__ __launch_bounds__(1024, 2) void batch_sparse_matrix_vector_multiplication(
    xpu::spmat_descript<devFdT> A, xpu::spmat_descript<devFdT> B, xpu::spmat_descript<devFdT> C, 
    devFdT *z, devFdT *Az, devFdT *Bz, devFdT *Cz) // NEED Az, Bz, Cz set to zero before calling
{
    size_t bid = blockIdx.x / 3, sel = blockIdx.x % 3;
    size_t tid = threadIdx.x + bid * blockDim.x, stride = gridDim.x * blockDim.x / 3;

    __shared__ devFdT shmem[1024];

    xpu::spmat_descript<devFdT> sd = (sel == 0) ? A : ((sel == 1) ? B : C);
    devFdT *dest = (sel == 0) ? Az : ((sel == 1) ? Bz : Cz);

    for (size_t i = tid; i < sd.row - 1; i += stride) {
        shmem[threadIdx.x].zero();
        for (size_t j = sd.row_ptr[i]; j < sd.row_ptr[i + 1]; j++) {
            shmem[threadIdx.x] += z[sd.col_idx[j]] * sd.val[j];
        }
        dest[i] = shmem[threadIdx.x];
    }
    if (tid == 0) {
        shmem[threadIdx.x].zero();    
        for (size_t j = sd.row_ptr[sd.row - 1]; j < sd.cnt; j++) shmem[threadIdx.x] += z[sd.col_idx[j]] * sd.val[j];
        dest[sd.row - 1] = shmem[threadIdx.x];
    }
}

template <typename devFdT> 
__global__ __launch_bounds__(1024, 2) void ntt_loop(size_t deg, devFdT *coeffs, devFdT *omega_powers, size_t hl, size_t i)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x, stride = blockDim.x * gridDim.x;
    for (size_t j = tid; j < deg; j += stride) if ((j & hl) == 0) {
        devFdT clo = coeffs[j], chi = coeffs[j ^ hl];
        size_t exp = ((j & (hl - 1)) << i);
        coeffs[j] = clo + chi; coeffs[j ^ hl] = (clo - chi) * omega_powers[exp];
    }
}
template <typename devFdT> 
__global__ __launch_bounds__(1024, 2) void ntt_bitrev_permutation(size_t deg, size_t log_deg, devFdT *coeffs)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x, stride = blockDim.x * gridDim.x;
    for (size_t j = tid; j < deg; j += stride)  {
        size_t rev_j = __brev(j) >> (32 - log_deg);
        if (j < rev_j) {
            devFdT tmp = coeffs[j];
            coeffs[j] = coeffs[rev_j];
            coeffs[rev_j] = tmp;
        }
    }
}
template <typename devFdT> 
__global__ __launch_bounds__(1024, 2) void ntt_mul_by_factor(size_t deg, devFdT *coeffs, devFdT *factor)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x, stride = blockDim.x * gridDim.x;
    __shared__ devFdT shfactor;
    if (threadIdx.x == 0) shfactor = *factor;
    __syncthreads();
    for (size_t j = tid; j < deg; j += stride) coeffs[j] *= shfactor;
}

template <typename devFdT>
void number_theory_transformation(size_t deg, size_t log_deg, devFdT *coeffs, devFdT *omega_powers, devFdT *factor)
{
    for (size_t hl = (deg >> 1), i = 0; hl > 0; hl >>= 1, i++) {
        (ntt_loop<devFdT>)<<<216, 1024>>>(deg, coeffs, omega_powers, hl, i);
    }
    (ntt_bitrev_permutation<devFdT>)<<<216, 1024>>>(deg, log_deg, coeffs);
    if (factor) (ntt_mul_by_factor<devFdT>)<<<216, 1024>>>(deg, coeffs, factor);
}

template <typename devFdT>
__global__ __launch_bounds__(1024, 2) void polynomial_A_times_B_minus_C(size_t deg, devFdT *A, devFdT *B, devFdT *C)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x, stride = blockDim.x * gridDim.x;
    __shared__ devFdT shmem[1024];
    for (size_t i = tid; i < deg; i += stride) {
        shmem[threadIdx.x] = A[i];
        shmem[threadIdx.x] *= B[i];
        shmem[threadIdx.x] -= C[i];
        A[i] = shmem[threadIdx.x];
    }
}