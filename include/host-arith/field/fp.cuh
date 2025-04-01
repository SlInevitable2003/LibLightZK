/** @file
 *****************************************************************************
 Declaration of arithmetic in the finite field F[p], for prime p of fixed length.
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#pragma once
#include "bigint.cuh"

namespace libff {

    template<size_t n, const bigint<n>& modulus>
    class Fp_model;

    /**
     * Arithmetic in the finite field F[p], for prime p of fixed length.
     *
     * This class implements Fp-arithmetic, for a large prime p, using a fixed number
     * of words. It is optimized for tight memory consumption, so the modulus p is
     * passed as a template parameter, to avoid per-element overheads.
     *
     * The implementation is mostly a wrapper around GMP's MPN (constant-size integers).
     * But for the integer sizes of interest for libff (3 to 5 limbs of 64 bits each),
     * we implement performance-critical routines, like addition and multiplication,
     * using hand-optimzied assembly code.
    */
    template<size_t n, const bigint<n>& modulus>
    class Fp_model {
    public:
        bigint<n> mont_repr;
    public:
        static const size_t num_limbs = n;
        static const constexpr bigint<n>& mod = modulus;

        static size_t num_bits;
        static bigint<n> euler; // (modulus-1)/2
        static size_t s; // modulus = 2^s * t + 1
        static bigint<n> t; // with t odd
        static bigint<n> t_minus_1_over_2; // (t-1)/2
        static Fp_model<n, modulus> nqr; // a quadratic nonresidue
        static Fp_model<n, modulus> nqr_to_t; // nqr^t
        static Fp_model<n, modulus> multiplicative_generator; // generator of Fp^*
        static Fp_model<n, modulus> root_of_unity; // generator^((modulus-1)/2^s)
        static uint64_t inv; // modulus^(-1) mod W, where W = 2^(word size)
        static bigint<n> Rsquared; // R^2, where R = W^k, where k = ??
        static bigint<n> Rcubed;   // R^3

        static bool modulus_is_valid() { return modulus.data[n-1] != 0; } // mpn inverse assumes that highest limb is non-zero

        Fp_model() {};
        Fp_model(const bigint<n> &b, const bool raw = true);
        Fp_model(const long x, const bool is_unsigned=false);

        void set_ulong(const unsigned long x);

        void mul_reduce(const bigint<n> &other);

        void clear();

        /* Return the standard (not Montgomery) representation of the
        Field element's requivalence class. I.e. Fp(2).as_bigint()
            would return bigint(2) */
        bigint<n> as_bigint() const;
        /* Return the last limb of the standard representation of the
        field element. E.g. on 64-bit architectures Fp(123).as_ulong()
        and Fp(2^64+123).as_ulong() would both return 123. */
        unsigned long as_ulong() const;

        bool operator==(const Fp_model& other) const;
        bool operator!=(const Fp_model& other) const;
        bool is_zero() const;

        void print() const;

        Fp_model& operator+=(const Fp_model& other);
        Fp_model& operator-=(const Fp_model& other);
        Fp_model& operator*=(const Fp_model& other);
        Fp_model& operator^=(const unsigned long pow);

        template<size_t m>
        Fp_model& operator^=(const bigint<m> &pow);

        Fp_model operator+(const Fp_model& other) const;
        Fp_model operator-(const Fp_model& other) const;
        Fp_model operator*(const Fp_model& other) const;
        Fp_model operator-() const;
        Fp_model squared() const;
        Fp_model& invert();
        Fp_model inverse() const;
        Fp_model sqrt() const; // HAS TO BE A SQUARE (else does not terminate)

        Fp_model operator^(const unsigned long pow) const;
        template<size_t m>
        Fp_model operator^(const bigint<m> &pow) const;

        static size_t size_in_bits() { return num_bits; }
        static size_t capacity() { return num_bits - 1; }
        static bigint<n> field_char() { return modulus; }

        static Fp_model<n, modulus> zero();
        static Fp_model<n, modulus> one();
        static Fp_model<n, modulus> random_element();
        static Fp_model<n, modulus> geometric_generator(); // generator^k, for k = 1 to m, domain size m
        static Fp_model<n, modulus> arithmetic_generator();// generator++, for k = 1 to m, domain size m
    };

    template<size_t n, const bigint<n>& modulus>
    size_t Fp_model<n, modulus>::num_bits;

    template<size_t n, const bigint<n>& modulus>
    bigint<n> Fp_model<n, modulus>::euler;

    template<size_t n, const bigint<n>& modulus>
    size_t Fp_model<n, modulus>::s;

    template<size_t n, const bigint<n>& modulus>
    bigint<n> Fp_model<n, modulus>::t;

    template<size_t n, const bigint<n>& modulus>
    bigint<n> Fp_model<n, modulus>::t_minus_1_over_2;

    template<size_t n, const bigint<n>& modulus>
    Fp_model<n, modulus> Fp_model<n, modulus>::nqr;

    template<size_t n, const bigint<n>& modulus>
    Fp_model<n, modulus> Fp_model<n, modulus>::nqr_to_t;

    template<size_t n, const bigint<n>& modulus>
    Fp_model<n, modulus> Fp_model<n, modulus>::multiplicative_generator;

    template<size_t n, const bigint<n>& modulus>
    Fp_model<n, modulus> Fp_model<n, modulus>::root_of_unity;

    template<size_t n, const bigint<n>& modulus>
    uint64_t Fp_model<n, modulus>::inv;

    template<size_t n, const bigint<n>& modulus>
    bigint<n> Fp_model<n, modulus>::Rsquared;

    template<size_t n, const bigint<n>& modulus>
    bigint<n> Fp_model<n, modulus>::Rcubed;

} // libff

/** @file
 *****************************************************************************
 Implementation of arithmetic in the finite field F[p], for prime p of fixed length.
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "fp_aux.cuh"
#include "exp.cuh"

namespace libff {

template<size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::mul_reduce(const bigint<n> &other)
{
    /* stupid pre-processor tricks; beware */
#if defined(__x86_64__)
    if (n == 3)
    { // Use asm-optimized Comba multiplication and reduction
        uint64_t res[2*n];
        uint64_t c0, c1, c2;
        COMBA_3_BY_3_MUL(c0, c1, c2, res, this->mont_repr.data, other.data);

        uint64_t k;
        uint64_t tmp1, tmp2, tmp3;
        REDUCE_6_LIMB_PRODUCT(k, tmp1, tmp2, tmp3, inv, res, modulus.data);

        /* subtract t > mod */
        __asm__
            ("/* check for overflow */        \n\t"
             MONT_CMP(16)
             MONT_CMP(8)
             MONT_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             MONT_FIRSTSUB
             MONT_NEXTSUB(8)
             MONT_NEXTSUB(16)
             "done%=:                         \n\t"
             :
             : [tmp] "r" (res+n), [M] "r" (modulus.data)
             : "cc", "memory", "%rax");
        mini_copyi(this->mont_repr.data, res+n, n);
    }
    else if (n == 4)
    { // use asm-optimized "CIOS method"

        uint64_t tmp[n+1];
        uint64_t T0=0, T1=1, cy=2, u=3; // TODO: fix this

        __asm__ (MONT_PRECOMPUTE
                 MONT_FIRSTITER(1)
                 MONT_FIRSTITER(2)
                 MONT_FIRSTITER(3)
                 MONT_FINALIZE(3)
                 MONT_ITERFIRST(1)
                 MONT_ITERITER(1, 1)
                 MONT_ITERITER(1, 2)
                 MONT_ITERITER(1, 3)
                 MONT_FINALIZE(3)
                 MONT_ITERFIRST(2)
                 MONT_ITERITER(2, 1)
                 MONT_ITERITER(2, 2)
                 MONT_ITERITER(2, 3)
                 MONT_FINALIZE(3)
                 MONT_ITERFIRST(3)
                 MONT_ITERITER(3, 1)
                 MONT_ITERITER(3, 2)
                 MONT_ITERITER(3, 3)
                 MONT_FINALIZE(3)
                 "/* check for overflow */        \n\t"
                 MONT_CMP(24)
                 MONT_CMP(16)
                 MONT_CMP(8)
                 MONT_CMP(0)

                 "/* subtract mod if overflow */  \n\t"
                 "subtract%=:                     \n\t"
                 MONT_FIRSTSUB
                 MONT_NEXTSUB(8)
                 MONT_NEXTSUB(16)
                 MONT_NEXTSUB(24)
                 "done%=:                         \n\t"
                 :
                 : [tmp] "r" (tmp), [A] "r" (this->mont_repr.data), [B] "r" (other.data), [inv] "r" (inv), [M] "r" (modulus.data),
                   [T0] "r" (T0), [T1] "r" (T1), [cy] "r" (cy), [u] "r" (u)
                 : "cc", "memory", "%rax", "%rdx"
        );
        mini_copyi(this->mont_repr.data, tmp, n);
    }
    else if (n == 5)
    { // use asm-optimized "CIOS method"

        uint64_t tmp[n+1];
        uint64_t T0=0, T1=1, cy=2, u=3; // TODO: fix this

        __asm__ (MONT_PRECOMPUTE
                 MONT_FIRSTITER(1)
                 MONT_FIRSTITER(2)
                 MONT_FIRSTITER(3)
                 MONT_FIRSTITER(4)
                 MONT_FINALIZE(4)
                 MONT_ITERFIRST(1)
                 MONT_ITERITER(1, 1)
                 MONT_ITERITER(1, 2)
                 MONT_ITERITER(1, 3)
                 MONT_ITERITER(1, 4)
                 MONT_FINALIZE(4)
                 MONT_ITERFIRST(2)
                 MONT_ITERITER(2, 1)
                 MONT_ITERITER(2, 2)
                 MONT_ITERITER(2, 3)
                 MONT_ITERITER(2, 4)
                 MONT_FINALIZE(4)
                 MONT_ITERFIRST(3)
                 MONT_ITERITER(3, 1)
                 MONT_ITERITER(3, 2)
                 MONT_ITERITER(3, 3)
                 MONT_ITERITER(3, 4)
                 MONT_FINALIZE(4)
                 MONT_ITERFIRST(4)
                 MONT_ITERITER(4, 1)
                 MONT_ITERITER(4, 2)
                 MONT_ITERITER(4, 3)
                 MONT_ITERITER(4, 4)
                 MONT_FINALIZE(4)
                 "/* check for overflow */        \n\t"
                 MONT_CMP(32)
                 MONT_CMP(24)
                 MONT_CMP(16)
                 MONT_CMP(8)
                 MONT_CMP(0)

                 "/* subtract mod if overflow */  \n\t"
                 "subtract%=:                     \n\t"
                 MONT_FIRSTSUB
                 MONT_NEXTSUB(8)
                 MONT_NEXTSUB(16)
                 MONT_NEXTSUB(24)
                 MONT_NEXTSUB(32)
                 "done%=:                         \n\t"
                 :
                 : [tmp] "r" (tmp), [A] "r" (this->mont_repr.data), [B] "r" (other.data), [inv] "r" (inv), [M] "r" (modulus.data),
                   [T0] "r" (T0), [T1] "r" (T1), [cy] "r" (cy), [u] "r" (u)
                 : "cc", "memory", "%rax", "%rdx"
        );
        mini_copyi(this->mont_repr.data, tmp, n);
    }
    else
#endif
    {
        uint64_t res[2*n];
        mini_mul_n(res, this->mont_repr.data, other.data, n);

        /*
          The Montgomery reduction here is based on Algorithm 14.32 in
          Handbook of Applied Cryptography
          <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
         */
        for (size_t i = 0; i < n; ++i)
        {
            uint64_t k = inv * res[i];
            /* calculate res = res + k * mod * b^i */
            uint64_t carryout = mini_addmul_1(res+i, modulus.data, n, k);
            carryout = mini_add_1(res+n+i, res+n+i, n-i, carryout);
            assert(carryout == 0);
        }

        if (mini_cmp(res+n, modulus.data, n) >= 0)
        {
            const uint64_t borrow = mini_sub(res+n, res+n, n, modulus.data, n);
            assert(borrow == 0);
        }

        mini_copyi(this->mont_repr.data, res+n, n);
    }
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>::Fp_model(const bigint<n> &b, const bool raw)
{
    if (!raw) mini_copyi(this->mont_repr.data, b.data, n);
    else {
        mini_copyi(this->mont_repr.data, Rsquared.data, n);
        mul_reduce(b);
    }
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>::Fp_model(const long x, const bool is_unsigned)
{
    static_assert(std::numeric_limits<uint64_t>::max() >= static_cast<unsigned long>(std::numeric_limits<long>::max()), "long won't fit in uint64_t");
    if (is_unsigned || x >= 0) this->mont_repr.data[0] = (uint64_t)x;
    else
    {
        const uint64_t borrow = mini_sub_1(this->mont_repr.data, modulus.data, n, (uint64_t)-x);
        assert(borrow == 0);
    }

    mul_reduce(Rsquared);
}

template<size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::set_ulong(const unsigned long x)
{
    this->mont_repr.clear();
    this->mont_repr.data[0] = x;
    mul_reduce(Rsquared);
}

template<size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::clear()
{
    this->mont_repr.clear();
}

template<size_t n, const bigint<n>& modulus>
bigint<n> Fp_model<n,modulus>::as_bigint() const
{
    bigint<n> one;
    one.clear();
    one.data[0] = 1;

    Fp_model<n, modulus> res(*this);
    res.mul_reduce(one);

    return (res.mont_repr);
}

template<size_t n, const bigint<n>& modulus>
unsigned long Fp_model<n,modulus>::as_ulong() const
{
    return this->as_bigint().as_ulong();
}

template<size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::operator==(const Fp_model& other) const
{
    return (this->mont_repr == other.mont_repr);
}

template<size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::operator!=(const Fp_model& other) const
{
    return (this->mont_repr != other.mont_repr);
}

template<size_t n, const bigint<n>& modulus>
bool Fp_model<n,modulus>::is_zero() const
{
    return (this->mont_repr.is_zero()); // zero maps to zero
}

template<size_t n, const bigint<n>& modulus>
void Fp_model<n,modulus>::print() const
{
    Fp_model<n,modulus> tmp;
    tmp.mont_repr.data[0] = 1;
    tmp.mul_reduce(this->mont_repr);

    tmp.mont_repr.print();
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::zero()
{
    Fp_model<n,modulus> res;
    res.mont_repr.clear();
    return res;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::one()
{
    Fp_model<n,modulus> res;
    res.mont_repr.data[0] = 1;
    res.mul_reduce(Rsquared);
    return res;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::geometric_generator()
{
    Fp_model<n,modulus> res;
    res.mont_repr.data[0] = 2;
    res.mul_reduce(Rsquared);
    return res;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::arithmetic_generator()
{
    Fp_model<n,modulus> res;
    res.mont_repr.data[0] = 1;
    res.mul_reduce(Rsquared);
    return res;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator+=(const Fp_model<n,modulus>& other)
{
#if defined(__x86_64__)
    if (n == 3)
    {
        __asm__
            ("/* perform bignum addition */   \n\t"
             ADD_FIRSTADD
             ADD_NEXTADD(8)
             ADD_NEXTADD(16)
             "/* if overflow: subtract     */ \n\t"
             "/* (tricky point: if A and B are in the range we do not need to do anything special for the possible carry flag) */ \n\t"
             "jc      subtract%=              \n\t"

             "/* check for overflow */        \n\t"
             ADD_CMP(16)
             ADD_CMP(8)
             ADD_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             ADD_FIRSTSUB
             ADD_NEXTSUB(8)
             ADD_NEXTSUB(16)
             "done%=:                         \n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 4)
    {
        __asm__
            ("/* perform bignum addition */   \n\t"
             ADD_FIRSTADD
             ADD_NEXTADD(8)
             ADD_NEXTADD(16)
             ADD_NEXTADD(24)
             "/* if overflow: subtract     */ \n\t"
             "/* (tricky point: if A and B are in the range we do not need to do anything special for the possible carry flag) */ \n\t"
             "jc      subtract%=              \n\t"

             "/* check for overflow */        \n\t"
             ADD_CMP(24)
             ADD_CMP(16)
             ADD_CMP(8)
             ADD_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             ADD_FIRSTSUB
             ADD_NEXTSUB(8)
             ADD_NEXTSUB(16)
             ADD_NEXTSUB(24)
             "done%=:                         \n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 5)
    {
        __asm__
            ("/* perform bignum addition */   \n\t"
             ADD_FIRSTADD
             ADD_NEXTADD(8)
             ADD_NEXTADD(16)
             ADD_NEXTADD(24)
             ADD_NEXTADD(32)
             "/* if overflow: subtract     */ \n\t"
             "/* (tricky point: if A and B are in the range we do not need to do anything special for the possible carry flag) */ \n\t"
             "jc      subtract%=              \n\t"

             "/* check for overflow */        \n\t"
             ADD_CMP(32)
             ADD_CMP(24)
             ADD_CMP(16)
             ADD_CMP(8)
             ADD_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             ADD_FIRSTSUB
             ADD_NEXTSUB(8)
             ADD_NEXTSUB(16)
             ADD_NEXTSUB(24)
             ADD_NEXTSUB(32)
             "done%=:                         \n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else
#endif
    {
        uint64_t scratch[n+1];
        const uint64_t carry = mini_add_n(scratch, this->mont_repr.data, other.mont_repr.data, n);
        scratch[n] = carry;

        if (carry || mini_cmp(scratch, modulus.data, n) >= 0)
        {
            const uint64_t borrow = mini_sub(scratch, scratch, n+1, modulus.data, n);
            assert(borrow == 0);
        }

        mini_copyi(this->mont_repr.data, scratch, n);
    }

    return *this;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator-=(const Fp_model<n,modulus>& other)
{
#if defined(__x86_64__)
    if (n == 3)
    {
        __asm__
            (SUB_FIRSTSUB
             SUB_NEXTSUB(8)
             SUB_NEXTSUB(16)

             "jnc     done%=\n\t"

             SUB_FIRSTADD
             SUB_NEXTADD(8)
             SUB_NEXTADD(16)

             "done%=:\n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 4)
    {
        __asm__
            (SUB_FIRSTSUB
             SUB_NEXTSUB(8)
             SUB_NEXTSUB(16)
             SUB_NEXTSUB(24)

             "jnc     done%=\n\t"

             SUB_FIRSTADD
             SUB_NEXTADD(8)
             SUB_NEXTADD(16)
             SUB_NEXTADD(24)

             "done%=:\n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else if (n == 5)
    {
        __asm__
            (SUB_FIRSTSUB
             SUB_NEXTSUB(8)
             SUB_NEXTSUB(16)
             SUB_NEXTSUB(24)
             SUB_NEXTSUB(32)

             "jnc     done%=\n\t"

             SUB_FIRSTADD
             SUB_NEXTADD(8)
             SUB_NEXTADD(16)
             SUB_NEXTADD(24)
             SUB_NEXTADD(32)

             "done%=:\n\t"
             :
             : [A] "r" (this->mont_repr.data), [B] "r" (other.mont_repr.data), [mod] "r" (modulus.data)
             : "cc", "memory", "%rax");
    }
    else
#endif
    {
        uint64_t scratch[n+1];
        if (mini_cmp(this->mont_repr.data, other.mont_repr.data, n) < 0)
        {
            const uint64_t carry = mini_add_n(scratch, this->mont_repr.data, modulus.data, n);
            scratch[n] = carry;
        }
        else
        {
            mini_copyi(scratch, this->mont_repr.data, n);
            scratch[n] = 0;
        }

        const uint64_t borrow = mini_sub(scratch, scratch, n+1, other.mont_repr.data, n);
        assert(borrow == 0);

        mini_copyi(this->mont_repr.data, scratch, n);
    }
    return *this;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator*=(const Fp_model<n,modulus>& other)
{
    mul_reduce(other.mont_repr);
    return *this;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator^=(const unsigned long pow)
{
    (*this) = power<Fp_model<n, modulus> >(*this, pow);
    return (*this);
}

template<size_t n, const bigint<n>& modulus>
template<size_t m>
Fp_model<n,modulus>& Fp_model<n,modulus>::operator^=(const bigint<m> &pow)
{
    (*this) = power<Fp_model<n, modulus>, m>(*this, pow);
    return (*this);
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator+(const Fp_model<n,modulus>& other) const
{
    Fp_model<n, modulus> r(*this);
    return (r += other);
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator-(const Fp_model<n,modulus>& other) const
{
    Fp_model<n, modulus> r(*this);
    return (r -= other);
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator*(const Fp_model<n,modulus>& other) const
{
    Fp_model<n, modulus> r(*this);
    return (r *= other);
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator^(const unsigned long pow) const
{
    Fp_model<n, modulus> r(*this);
    return (r ^= pow);
}

template<size_t n, const bigint<n>& modulus>
template<size_t m>
Fp_model<n,modulus> Fp_model<n,modulus>::operator^(const bigint<m> &pow) const
{
    Fp_model<n, modulus> r(*this);
    return (r ^= pow);
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::operator-() const
{
    if (this->is_zero())
    {
        return (*this);
    }
    else
    {
        Fp_model<n, modulus> r;
        mini_sub_n(r.mont_repr.data, modulus.data, this->mont_repr.data, n);
        return r;
    }
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::squared() const
{
#if defined(__x86_64__)
    if (n == 3)
    { // use asm-optimized Comba squaring
        uint64_t res[2*n];
        uint64_t c0, c1, c2;
        COMBA_3_BY_3_SQR(c0, c1, c2, res, this->mont_repr.data);

        uint64_t k;
        uint64_t tmp1, tmp2, tmp3;
        REDUCE_6_LIMB_PRODUCT(k, tmp1, tmp2, tmp3, inv, res, modulus.data);

        /* subtract t > mod */
        __asm__ volatile
            ("/* check for overflow */        \n\t"
             MONT_CMP(16)
             MONT_CMP(8)
             MONT_CMP(0)

             "/* subtract mod if overflow */  \n\t"
             "subtract%=:                     \n\t"
             MONT_FIRSTSUB
             MONT_NEXTSUB(8)
             MONT_NEXTSUB(16)
             "done%=:                         \n\t"
             :
             : [tmp] "r" (res+n), [M] "r" (modulus.data)
             : "cc", "memory", "%rax");

        Fp_model<n, modulus> r;
        mini_copyi(r.mont_repr.data, res+n, n);
        return r;
    }
    else
#endif
    {
        Fp_model<n, modulus> r(*this);
        return (r *= r);
    }
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus>& Fp_model<n,modulus>::invert()
{
    assert(!this->is_zero());
    bigint<n> modulus_minus_2;
    mini_sub_1(modulus_minus_2.data, modulus.data, n, 2);
    
    Fp_model<n, modulus> r = Fp_model<n, modulus>::one();
    for (size_t i = n * MINI_NUMB_BITS; i > 0; i--) {
        r = r.squared();
        if (modulus_minus_2.test_bit(i - 1)) r *= (*this);
    }
    (*this) = r;
    return *this;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::inverse() const
{
    Fp_model<n, modulus> r(*this);
    return (r.invert());
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n, modulus> Fp_model<n,modulus>::random_element() /// returns random element of Fp_model
{
    /* note that as Montgomery representation is a bijection then
       selecting a random element of {xR} is the same as selecting a
       random element of {x} */
    Fp_model<n, modulus> r;
    do
    {
        r.mont_repr.randomize();

        /* clear all bits higher than MSB of modulus */
        size_t bitno = MINI_NUMB_BITS * n - 1;
        while (modulus.test_bit(bitno) == false)
        {
            const std::size_t part = bitno/MINI_NUMB_BITS;
            const std::size_t bit = bitno - (MINI_NUMB_BITS*part);

            r.mont_repr.data[part] &= ~(1ul<<bit);

            bitno--;
        }
    }
   /* if r.data is still >= modulus -- repeat (rejection sampling) */
    while (mini_cmp(r.mont_repr.data, modulus.data, n) >= 0);

    return r;
}

template<size_t n, const bigint<n>& modulus>
Fp_model<n,modulus> Fp_model<n,modulus>::sqrt() const
{
    Fp_model<n,modulus> one = Fp_model<n,modulus>::one();

    size_t v = Fp_model<n,modulus>::s;
    Fp_model<n,modulus> z = Fp_model<n,modulus>::nqr_to_t;
    Fp_model<n,modulus> w = (*this)^Fp_model<n,modulus>::t_minus_1_over_2;
    Fp_model<n,modulus> x = (*this) * w;
    Fp_model<n,modulus> b = x * w; // b = (*this)^t

    // compute square root with Tonelli--Shanks
    // (does not terminate if not a square!)

    while (b != one)
    {
        size_t m = 0;
        Fp_model<n,modulus> b2m = b;
        while (b2m != one)
        {
            /* invariant: b2m = b^(2^m) after entering this loop */
            b2m = b2m.squared();
            m += 1;
        }

        int j = v-m-1;
        w = z;
        while (j > 0)
        {
            w = w.squared();
            --j;
        } // w = z^2^(v-m-1)

        z = w.squared();
        b = b * z;
        x = x * w;
        v = m;
    }

    return x;
}

} // libff