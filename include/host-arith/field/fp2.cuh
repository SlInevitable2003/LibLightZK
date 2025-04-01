/** @file
 *****************************************************************************
 Implementation of arithmetic in the finite field F[p^2].
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#pragma once
#include <vector>
#include "fp.cuh"

namespace libff {

    template<size_t n, const bigint<n>& modulus>
    class Fp2_model;

    /**
     * Arithmetic in the field F[p^2].
     *
     * Let p := modulus. This interface provides arithmetic for the extension field
     * Fp2 = Fp[U]/(U^2-non_residue), where non_residue is in Fp.
     *
     * ASSUMPTION: p = 1 (mod 6)
     */
    template<size_t n, const bigint<n>& modulus>
    class Fp2_model {
    public:
        typedef Fp_model<n, modulus> my_Fp;

        static bigint<2*n> euler; // (modulus^2-1)/2
        static size_t s;       // modulus^2 = 2^s * t + 1
        static bigint<2*n> t;  // with t odd
        static bigint<2*n> t_minus_1_over_2; // (t-1)/2
        static my_Fp non_residue; // X^4-non_residue irreducible over Fp; used for constructing Fp2 = Fp[X] / (X^2 - non_residue)
        static Fp2_model<n, modulus> nqr; // a quadratic nonresidue in Fp2
        static Fp2_model<n, modulus> nqr_to_t; // nqr^t
        static my_Fp Frobenius_coeffs_c1[2]; // non_residue^((modulus^i-1)/2) for i=0,1

        my_Fp c0, c1;
        Fp2_model() {};
        Fp2_model(const my_Fp& c0, const my_Fp& c1) : c0(c0), c1(c1) {};

        void clear() { c0.clear(); c1.clear(); }
        void print() const { printf("c0/c1:\n"); c0.print(); c1.print(); }

        static Fp2_model<n, modulus> zero();
        static Fp2_model<n, modulus> one();
        static Fp2_model<n, modulus> random_element();

        bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
        bool operator==(const Fp2_model &other) const;
        bool operator!=(const Fp2_model &other) const;

        Fp2_model operator+(const Fp2_model &other) const;
        Fp2_model operator-(const Fp2_model &other) const;
        Fp2_model operator*(const Fp2_model &other) const;
        Fp2_model operator-() const;
        Fp2_model squared() const; // default is squared_complex
        Fp2_model inverse() const;
        Fp2_model Frobenius_map(unsigned long power) const;
        Fp2_model sqrt() const; // HAS TO BE A SQUARE (else does not terminate)
        Fp2_model squared_karatsuba() const;
        Fp2_model squared_complex() const;

        template<size_t m>
        Fp2_model operator^(const bigint<m> &other) const;

        static size_t size_in_bits() { return 2*my_Fp::size_in_bits(); }
        static bigint<n> base_field_char() { return modulus; }
    };

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n, modulus> operator*(const Fp_model<n, modulus> &lhs, const Fp2_model<n, modulus> &rhs);

    template<size_t n, const bigint<n>& modulus>
    bigint<2*n> Fp2_model<n, modulus>::euler;

    template<size_t n, const bigint<n>& modulus>
    size_t Fp2_model<n, modulus>::s;

    template<size_t n, const bigint<n>& modulus>
    bigint<2*n> Fp2_model<n, modulus>::t;

    template<size_t n, const bigint<n>& modulus>
    bigint<2*n> Fp2_model<n, modulus>::t_minus_1_over_2;

    template<size_t n, const bigint<n>& modulus>
    Fp_model<n, modulus> Fp2_model<n, modulus>::non_residue;

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n, modulus> Fp2_model<n, modulus>::nqr;

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n, modulus> Fp2_model<n, modulus>::nqr_to_t;

    template<size_t n, const bigint<n>& modulus>
    Fp_model<n, modulus> Fp2_model<n, modulus>::Frobenius_coeffs_c1[2];

} // libff

/** @file
 *****************************************************************************
 Implementation of arithmetic in the finite field F[p^2].
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#include "field_utils.cuh"

namespace libff {

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::zero()
    {
        return Fp2_model<n, modulus>(my_Fp::zero(), my_Fp::zero());
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::one()
    {
        return Fp2_model<n, modulus>(my_Fp::one(), my_Fp::zero());
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::random_element()
    {
        Fp2_model<n, modulus> r;
        r.c0 = my_Fp::random_element();
        r.c1 = my_Fp::random_element();

        return r;
    }

    template<size_t n, const bigint<n>& modulus>
    bool Fp2_model<n,modulus>::operator==(const Fp2_model<n,modulus> &other) const
    {
        return (this->c0 == other.c0 && this->c1 == other.c1);
    }

    template<size_t n, const bigint<n>& modulus>
    bool Fp2_model<n,modulus>::operator!=(const Fp2_model<n,modulus> &other) const
    {
        return !(operator==(other));
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::operator+(const Fp2_model<n,modulus> &other) const
    {
        return Fp2_model<n,modulus>(this->c0 + other.c0,
                                    this->c1 + other.c1);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::operator-(const Fp2_model<n,modulus> &other) const
    {
        return Fp2_model<n,modulus>(this->c0 - other.c0,
                                    this->c1 - other.c1);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n, modulus> operator*(const Fp_model<n, modulus> &lhs, const Fp2_model<n, modulus> &rhs)
    {
        return Fp2_model<n,modulus>(lhs*rhs.c0,
                                    lhs*rhs.c1);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::operator*(const Fp2_model<n,modulus> &other) const
    {
        /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Karatsuba) */
        const my_Fp
            &A = other.c0, &B = other.c1,
            &a = this->c0, &b = this->c1;
        const my_Fp aA = a * A;
        const my_Fp bB = b * B;

        return Fp2_model<n,modulus>(aA + non_residue * bB,
                                    (a + b)*(A+B) - aA - bB);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::operator-() const
    {
        return Fp2_model<n,modulus>(-this->c0,
                                    -this->c1);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::squared() const
    {
        return squared_complex();
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::squared_karatsuba() const
    {
        /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Karatsuba squaring) */
        const my_Fp &a = this->c0, &b = this->c1;
        const my_Fp asq = a.squared();
        const my_Fp bsq = b.squared();

        return Fp2_model<n,modulus>(asq + non_residue * bsq,
                                    (a + b).squared() - asq - bsq);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::squared_complex() const
    {
        /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Complex squaring) */
        const my_Fp &a = this->c0, &b = this->c1;
        const my_Fp ab = a * b;

        return Fp2_model<n,modulus>((a + b) * (a + non_residue * b) - ab - non_residue * ab,
                                    ab + ab);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::inverse() const
    {
        const my_Fp &a = this->c0, &b = this->c1;

        /* From "High-Speed Software Implementation of the Optimal Ate Pairing over Barreto-Naehrig Curves"; Algorithm 8 */
        const my_Fp t0 = a.squared();
        const my_Fp t1 = b.squared();
        const my_Fp t2 = t0 - non_residue * t1;
        const my_Fp t3 = t2.inverse();
        const my_Fp c0 = a * t3;
        const my_Fp c1 = - (b * t3);

        return Fp2_model<n,modulus>(c0, c1);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::Frobenius_map(unsigned long power) const
    {
        return Fp2_model<n,modulus>(c0,
                                    Frobenius_coeffs_c1[power % 2] * c1);
    }

    template<size_t n, const bigint<n>& modulus>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::sqrt() const
    {
        Fp2_model<n,modulus> one = Fp2_model<n,modulus>::one();

        size_t v = Fp2_model<n,modulus>::s;
        Fp2_model<n,modulus> z = Fp2_model<n,modulus>::nqr_to_t;
        Fp2_model<n,modulus> w = (*this)^Fp2_model<n,modulus>::t_minus_1_over_2;
        Fp2_model<n,modulus> x = (*this) * w;
        Fp2_model<n,modulus> b = x * w; // b = (*this)^t

        // compute square root with Tonelli--Shanks
        // (does not terminate if not a square!)

        while (b != one)
        {
            size_t m = 0;
            Fp2_model<n,modulus> b2m = b;
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

    template<size_t n, const bigint<n>& modulus>
    template<size_t m>
    Fp2_model<n,modulus> Fp2_model<n,modulus>::operator^(const bigint<m> &pow) const
    {
        return power<Fp2_model<n, modulus>, m>(*this, pow);
    }

} // libff