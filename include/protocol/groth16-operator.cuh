#pragma once
#include "ntt-params.cuh"
#include "xpu-spmat.cuh"
#include "device-utils.cuh"

template <typename FieldT, ntt_config *nc>
void ntt_setup(size_t M, xpu::vector<FieldT>& ntt_params)
{
    ntt_params.allocate(6 * M + 2, xpu::mem_policy::cross_platform);
    size_t int_order = __builtin_ctzl(M), eval_order = __builtin_ctzl(M) + 1;
    ntt_params[0] = ntt_params[M] = ntt_params[2*M+1] = ntt_params[4*M+1] = FieldT::one();
    ntt_params[2*M] = FieldT(nc->domain_scale_inverse + int_order * nc->str_len, false);
    ntt_params[6*M+1] = FieldT(nc->domain_scale_inverse + (eval_order + 1) * nc->str_len, false);
    FieldT int_omega(nc->forward_root_of_unity + int_order * nc->str_len, false),
            int_omega_inv(nc->inverse_root_of_unity + int_order * nc->str_len, false),
            eval_omega(nc->forward_root_of_unity + eval_order * nc->str_len, false),
            eval_omega_inv(nc->inverse_root_of_unity + eval_order * nc->str_len, false);
    assert(int_omega * int_omega_inv == FieldT::one() && eval_omega * eval_omega_inv == FieldT::one());
    for (size_t i = 1; i < M; i++) {
        ntt_params[i] = ntt_params[i - 1] * int_omega;
        ntt_params[M + i] = ntt_params[M + i - 1] * int_omega_inv;
        assert(ntt_params[i] != FieldT::one() && ntt_params[M + i] != FieldT::one());
    }
    assert(ntt_params[1] * ntt_params[M - 1] == FieldT::one() && ntt_params[M + 1] * ntt_params[2 * M - 1] == FieldT::one());
    for (size_t i = 1; i < 2 * M; i++) {
        ntt_params[2 * M + 1 + i] = ntt_params[2 * M + i] * eval_omega;
        ntt_params[4 * M + 1 + i] = ntt_params[4 * M + i] * eval_omega_inv;
        assert(ntt_params[2 * M + 1 + i] != FieldT::one() && ntt_params[4 * M + 1 + i] != FieldT::one());
    }
    assert(ntt_params[2 * M + 2] * ntt_params[4 * M] == FieldT::one() && ntt_params[4 * M + 2] * ntt_params[6 * M] == FieldT::one());
    ntt_params.store();
}

#include "msm.cuh"
#include "poly.cuh"
#include "light-msm.cuh"