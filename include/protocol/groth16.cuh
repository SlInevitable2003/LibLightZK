#pragma once
#include "xpu-spmat.cuh"
#include "device-utils.cuh"
#include "timer.cuh"

#include "groth16-operator.cuh"

#include "ntt-params.cuh"
#include "pippenger-config.cuh"

enum class input_scale {
    // For MatMul: k = 3p^2, N = 3p^2 + p^3, M = next_pow2(p^2 + p^3)
    MatMul_p_39,
    MatMul_p_63,
    MatMul_p_100,
};

template <typename FieldT, typename G1T, typename G2T,
          typename devFdT, typename dG1, typename dG2,
          ntt_config* nc>
class Groth16 {
    size_t k, N, M;
    xpu::vector<FieldT> z;
    xpu::spmat<FieldT> A, B, C;

    input_scale is;

    xpu::vector<G1T> alpha1, beta1, delta1, a1, b1, Kpk, Z1;
    xpu::vector<G2T> beta2, delta2, b2;

    xpu::vector<FieldT> xA, xB, xC;
    //            0                           M-1 M                             2M-1 2M       2M+1                                4M 4M+1                                   6M 6M+1
    // ntt_params [omega_M^0, ..., omega_M^(M-1)] [omega_M^0, ..., omega_M^(-(M-1))] [M^(-1)] [omega_(2M)^0, ..., omega_(2M)^(2M-1)] [omega_(2M)^0, ..., omega_(2M)^(-(2M-1))] [(2M)^(-1)]
    xpu::vector<FieldT> ntt_params;

    pippenger_config pc;

public:
    Groth16(input_scale is) : is(is)
    {
        switch (is) {
            case input_scale::MatMul_p_39: { k = 4563, N = 63882, M = 65536; } break; // 2^16
            case input_scale::MatMul_p_63: { k = 11907, N = 261954, M = 262144; } break; // 2^18
            case input_scale::MatMul_p_100: { k = 30000, N = 1030000, M = 1048576; } break; // 2^20
        }
    }

    void Setup(size_t win_siz)
    {
        Timer timer;

        pc = {win_siz, (8 * sizeof(FieldT) + (win_siz - 1)) / win_siz, (size_t(1) << win_siz)};

        timer.start();
        switch (is) {
            case input_scale::MatMul_p_39: {
                A.init(xpu::application_scence::MatMul, xpu::matrix_type::A, 39);
                B.init(xpu::application_scence::MatMul, xpu::matrix_type::B, 39);
                C.init(xpu::application_scence::MatMul, xpu::matrix_type::C, 39);
            } break;
            case input_scale::MatMul_p_63: {
                A.init(xpu::application_scence::MatMul, xpu::matrix_type::A, 63);
                B.init(xpu::application_scence::MatMul, xpu::matrix_type::B, 63);
                C.init(xpu::application_scence::MatMul, xpu::matrix_type::C, 63);
            } break;
            case input_scale::MatMul_p_100: {
                A.init(xpu::application_scence::MatMul, xpu::matrix_type::A, 100);
                B.init(xpu::application_scence::MatMul, xpu::matrix_type::B, 100);
                C.init(xpu::application_scence::MatMul, xpu::matrix_type::C, 100);
            } break;
        }
        CUDA_DEBUG;
        timer.stop("Matrix Setup");
        
        timer.start();
        ntt_setup<FieldT, nc>(M, ntt_params);
        CUDA_DEBUG;
        timer.stop("NTT Setup");

        timer.start();
        int dev_cnt;
        cudaGetDeviceCount(&dev_cnt);

        alpha1.allocate(1, xpu::mem_policy::cross_platform); beta1.allocate(1, xpu::mem_policy::cross_platform); delta1.allocate(1, xpu::mem_policy::cross_platform);
        beta2.allocate(1, xpu::mem_policy::cross_platform); delta2.allocate(1, xpu::mem_policy::cross_platform);
        alpha1[0] = G1T::random_element(), beta1[0] = G1T::random_element(), delta1[0] = G1T::random_element();
        beta2[0] = G2T::random_element(), delta2[0] = G2T::random_element();
        alpha1.store(); beta1.store(); delta1.store();
        beta2.store(); delta2.store();

        xpu::vector<FieldT> ss(max(N + 1, M - 1), xpu::mem_policy::cross_platform);
        
        a1.allocate(pc.win_cnt * (N + 1), xpu::mem_policy::device_only);
        #pragma omp parallel for
        for (size_t i = 0; i < N + 1; i++) ss[i] = FieldT::random_element();
        ss.store();
        #pragma omp parallel for
        (fix_base_multi_scalar_multiplication_g1<devFdT, dG1>)<<<324, 510>>>((dG1*)a1.p(), (devFdT*)ss.p(), N + 1);
        
        b1.allocate(pc.win_cnt * (N + 1), xpu::mem_policy::device_only);
        #pragma omp parallel for
        for (size_t i = 0; i < N + 1; i++) ss[i] = FieldT::random_element();
        ss.store();
        (fix_base_multi_scalar_multiplication_g1<devFdT, dG1>)<<<324, 510>>>((dG1*)b1.p(), (devFdT*)ss.p(), N + 1);

        Kpk.allocate(pc.win_cnt * (N - k), xpu::mem_policy::device_only);
        #pragma omp parallel for
        for (size_t i = 0; i < N - k; i++) ss[i] = FieldT::random_element();
        ss.store();
        (fix_base_multi_scalar_multiplication_g1<devFdT, dG1>)<<<324, 510>>>((dG1*)Kpk.p(), (devFdT*)ss.p(), N - k);

        Z1.allocate(pc.win_cnt * (M - 1), xpu::mem_policy::device_only);
        #pragma omp parallel for
        for (size_t i = 0; i < M - 1; i++) ss[i] = FieldT::random_element();
        ss.store();
        (fix_base_multi_scalar_multiplication_g1<devFdT, dG1>)<<<324, 510>>>((dG1*)Z1.p(), (devFdT*)ss.p(), M - 1);

        b2.allocate(pc.win_cnt * (N + 1), xpu::mem_policy::device_only);
        #pragma omp parallel for
        for (size_t i = 0; i < N + 1; i++) ss[i] = FieldT::random_element();
        ss.store();
        (fix_base_multi_scalar_multiplication_g2<devFdT, dG2>)<<<324, 510>>>((dG2*)b2.p(), (devFdT*)ss.p(), N + 1);

        CUDA_DEBUG;
        timer.stop("Proving Key Setup");

        timer.start();
        z.allocate(1 + N, xpu::mem_policy::cross_platform);
        z[0] = FieldT::one();
        #pragma omp parallel for
        for (size_t i = 1; i <= N; i++) z[i] = FieldT::random_element();
        z.store();
        CUDA_DEBUG;
        timer.stop("Input Setup");

        timer.start();
        xA.allocate(2 * M, xpu::mem_policy::cross_platform), xB.allocate(2 * M, xpu::mem_policy::cross_platform), xC.allocate(2 * M, xpu::mem_policy::cross_platform);
        #pragma omp parallel for
        for (size_t i = 0; i < 2 * M; i++) xA[i] = xB[i] = xC[i] = FieldT::zero();
        xA.store(), xB.store(), xC.store();
        CUDA_DEBUG;
        timer.stop("Memory Prepare");

        timer.start();
        (pre_comp_g1<dG1>)<<<216, 1024>>>(N + 1, (dG1*)a1.p(), pc.win_siz, pc.win_cnt);
        (pre_comp_g1<dG1>)<<<216, 1024>>>(N + 1, (dG1*)b1.p(), pc.win_siz, pc.win_cnt);
        (pre_comp_g1<dG1>)<<<216, 1024>>>(N - k, (dG1*)Kpk.p(), pc.win_siz, pc.win_cnt);
        (pre_comp_g1<dG1>)<<<216, 1024>>>(M - 1, (dG1*)Z1.p(), pc.win_siz, pc.win_cnt);
        (pre_comp_g2<dG2>)<<<216, 1024>>>(N + 1, (dG2*)b2.p(), pc.win_siz, pc.win_cnt);
        CUDA_DEBUG;
        timer.stop("Pre-Calculation");
    }

    void Prove() 
    {
        Timer timer;

        CUDA_DEBUG;
        timer.start();
        (batch_sparse_matrix_vector_multiplication<devFdT>)<<<216, 1024>>>(
            A.template descript<devFdT>(), B.template descript<devFdT>(), C.template descript<devFdT>(), 
            (devFdT*)z.p(), (devFdT*)xA.p(), (devFdT*)xB.p(), (devFdT*)xC.p());
        CUDA_DEBUG;
        timer.stop("Phase 1: Sparse Matrix-Vector Multiplication");

        CUDA_DEBUG;
        timer.start();
        number_theory_transformation<devFdT>(M, __builtin_ctzl(M), (devFdT*)xA.p(), (devFdT*)ntt_params.p() + M, (devFdT*)ntt_params.p() + 2*M);
        number_theory_transformation<devFdT>(M, __builtin_ctzl(M), (devFdT*)xB.p(), (devFdT*)ntt_params.p() + M, (devFdT*)ntt_params.p() + 2*M);
        number_theory_transformation<devFdT>(M, __builtin_ctzl(M), (devFdT*)xC.p(), (devFdT*)ntt_params.p() + M, (devFdT*)ntt_params.p() + 2*M);
        CUDA_DEBUG;
        timer.stop("Phase 2: Inverse Number-Theory-Transformation for Interpolation");

        CUDA_DEBUG;
        timer.start();
        number_theory_transformation<devFdT>(2*M, __builtin_ctzl(2*M), (devFdT*)xA.p(), (devFdT*)ntt_params.p() + 2*M+1, 0);
        number_theory_transformation<devFdT>(2*M, __builtin_ctzl(2*M), (devFdT*)xB.p(), (devFdT*)ntt_params.p() + 2*M+1, 0);
        number_theory_transformation<devFdT>(2*M, __builtin_ctzl(2*M), (devFdT*)xC.p(), (devFdT*)ntt_params.p() + 2*M+1, 0);
        CUDA_DEBUG;
        timer.stop("Phase 3: Number-Theory-Transformation for Arithmetic");

        CUDA_DEBUG;
        timer.start();
        (polynomial_A_times_B_minus_C<devFdT>)<<<216, 1024>>>(2*M, (devFdT*)xA.p(), (devFdT*)xB.p(), (devFdT*)xC.p());
        number_theory_transformation<devFdT>(2*M, __builtin_ctzl(2*M), (devFdT*)xA.p(), (devFdT*)ntt_params.p() + 4*M+1, (devFdT*)ntt_params.p() + 6*M+1);
        CUDA_DEBUG;
        timer.stop("Phase 4: Component-wise Arithmetic and Inverse Number-Theory-Transformation for Restore");

        xpu::vector<G1T> g1res(4, xpu::mem_policy::cross_platform);
        xpu::vector<G2T> g2res(1, xpu::mem_policy::cross_platform);
        FieldT r = FieldT::random_element(), s = FieldT::random_element();

        CUDA_DEBUG;
        timer.start();
        multi_scalar_multiplication_g1<devFdT, dG1>(N+1, (devFdT*)z.p(), (dG1*)a1.p(), (dG1*)g1res.p(), pc.win_siz, pc.win_cnt, pc.bucket_cnt, N+1);
        multi_scalar_multiplication_g1<devFdT, dG1>(N+1, (devFdT*)z.p(), (dG1*)b1.p(), (dG1*)g1res.p()+1, pc.win_siz, pc.win_cnt, pc.bucket_cnt, N+1);
        multi_scalar_multiplication_g1<devFdT, dG1>(N-k, (devFdT*)z.p()+k+1, (dG1*)Kpk.p(), (dG1*)g1res.p()+2, pc.win_siz, pc.win_cnt, pc.bucket_cnt, N-k);
        multi_scalar_multiplication_g1<devFdT, dG1>(M-1, (devFdT*)xA.p()+M+1, (dG1*)Z1.p(), (dG1*)g1res.p()+3, pc.win_siz, pc.win_cnt, pc.bucket_cnt, M-1);
        CUDA_DEBUG;
        timer.stop("Phase 5-1: Multi-Scalar-Multiplication for G1 Group");

        CUDA_DEBUG;
        timer.start();
        multi_scalar_multiplication_g2<devFdT, dG2>(N+1, (devFdT*)z.p(), (dG2*)b2.p(), (dG2*)g2res.p(), pc.win_siz, pc.win_cnt, pc.bucket_cnt, N+1);
        CUDA_DEBUG;
        timer.stop("Phase 5-2: Multi-Scalar-Multiplication for G2 Group");

        CUDA_DEBUG;
        timer.start();
        g1res.load(); g2res.load();
        G1T Ar = alpha1[0] + g1res[0] + r * delta1[0],
            Bs1 = beta1[0] + g1res[1] + s * delta1[0];
        G1T Krs = s * Ar + r * Bs1 + ((-r) * s) * delta1[0] + g1res[2],
            HZ = g1res[3];
        G2T Bs2 = beta2[0] + g2res[0] + s * delta2[0];
        CUDA_DEBUG;
        timer.stop("Phase 6: Return the Proof");

        Ar.print();
        (Krs + HZ).print();
        Bs2.print();
    }
};

#include "alt_bn128_t.cuh"
#include "alt_bn128_g1.cuh"
#include "alt_bn128_g2.cuh"

typedef Groth16<libff::alt_bn128_Fr, libff::alt_bn128_G1, libff::alt_bn128_G2, alt_bn128::fr_t, alt_bn128::g1_t, alt_bn128::g2_t, &alt_bn128_ntt_config> Groth16_BN254;