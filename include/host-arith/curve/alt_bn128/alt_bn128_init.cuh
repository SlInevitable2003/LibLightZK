/** @file
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#pragma once
#include "fp.cuh"
#include "fp2.cuh"

namespace libff {

    const size_t alt_bn128_r_bitcount = 254;
    const size_t alt_bn128_q_bitcount = 254;

    const size_t alt_bn128_r_limbs = (alt_bn128_r_bitcount+MINI_NUMB_BITS-1)/MINI_NUMB_BITS;
    const size_t alt_bn128_q_limbs = (alt_bn128_q_bitcount+MINI_NUMB_BITS-1)/MINI_NUMB_BITS;

    extern bigint<alt_bn128_r_limbs> alt_bn128_modulus_r;
    extern bigint<alt_bn128_q_limbs> alt_bn128_modulus_q;

    typedef Fp_model<alt_bn128_r_limbs, alt_bn128_modulus_r> alt_bn128_Fr;
    typedef Fp_model<alt_bn128_q_limbs, alt_bn128_modulus_q> alt_bn128_Fq;
    typedef Fp2_model<alt_bn128_q_limbs, alt_bn128_modulus_q> alt_bn128_Fq2;
    // typedef Fp6_3over2_model<alt_bn128_q_limbs, alt_bn128_modulus_q> alt_bn128_Fq6;
    // typedef Fp12_2over3over2_model<alt_bn128_q_limbs, alt_bn128_modulus_q> alt_bn128_Fq12;
    // typedef alt_bn128_Fq12 alt_bn128_GT;

    // parameters for Barreto--Naehrig curve E/Fq : y^2 = x^3 + b
    extern alt_bn128_Fq alt_bn128_coeff_b;
    // parameters for twisted Barreto--Naehrig curve E'/Fq2 : y^2 = x^3 + b/xi
    extern alt_bn128_Fq2 alt_bn128_twist;
    extern alt_bn128_Fq2 alt_bn128_twist_coeff_b;
    extern alt_bn128_Fq alt_bn128_twist_mul_by_b_c0;
    extern alt_bn128_Fq alt_bn128_twist_mul_by_b_c1;
    extern alt_bn128_Fq2 alt_bn128_twist_mul_by_q_X;
    extern alt_bn128_Fq2 alt_bn128_twist_mul_by_q_Y;

    // parameters for pairing
    extern bigint<alt_bn128_q_limbs> alt_bn128_ate_loop_count;
    extern bool alt_bn128_ate_is_loop_count_neg;
    extern bigint<12*alt_bn128_q_limbs> alt_bn128_final_exponent;
    extern bigint<alt_bn128_q_limbs> alt_bn128_final_exponent_z;
    extern bool alt_bn128_final_exponent_is_z_neg;

    void init_alt_bn128_params();

    class alt_bn128_G1;
    class alt_bn128_G2;

} // libff
