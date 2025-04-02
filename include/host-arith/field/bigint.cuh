/** @file
 *****************************************************************************
 Declaration of bigint wrapper class around GMP's MPZ long integers.
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#pragma once
#include <cstddef>
#include <iostream>
#include "host-def.cuh"

namespace libff {

    template<size_t n> class bigint;

    /**
    * Wrapper class around GMP's MPZ long integers. It supports arithmetic operations,
    * serialization and randomization. Serialization is fragile, see common/serialization.hpp.
    */

    template<size_t n>
    class bigint {
    public:
        static const size_t N = n;

        uint64_t data[n] = {0};

        bigint() = default;
        bigint(const uint64_t x); /// Initalize from a small integer
        bigint(const char* s); /// Initialize from a string containing an integer in hex notation

        void print() const;
        bool operator==(const bigint<n>& other) const;
        bool operator!=(const bigint<n>& other) const;
        void clear();
        bool is_zero() const;
        size_t max_bits() const { return n * MINI_NUMB_BITS; } /// Returns the number of bits representable by this bigint type
        size_t num_bits() const; /// Returns the number of bits in this specific bigint value, i.e., position of the most-significant 1

        uint64_t as_ulong() const; /// Return the last limb of the integer
        bool test_bit(const std::size_t bitno) const;

        bigint& randomize();
};

} // libff

/** @file
 *****************************************************************************
 Implementation of bigint wrapper class around GMP's MPZ long integers.
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#include <cassert>
#include <cstring>
#include <random>
#include "mini-gmp.cuh"

namespace libff {

    template<size_t n>
    bigint<n>::bigint(const unsigned long x) /// Initalize from a small integer
    {
        assert(8*sizeof(x) <= MINI_NUMB_BITS);
        this->data[0] = x;
    }

    template<size_t n>
    bigint<n>::bigint(const char* s) /// Initialize from a string containing an integer in decimal notation
    {
        size_t l = strlen(s);
        unsigned char* s_copy = new unsigned char[l];

        for (size_t i = 0; i < l; ++i) {
            if (s[i] >= '0' && s[i] <= '9') s_copy[i] = s[i] - '0';
            else if (s[i] >= 'a' && s[i] <= 'f') s_copy[i] = 10 + (s[i] - 'a');
            else if (s[i] >= 'A' && s[i] <= 'F') s_copy[i] = 10 + (s[i] - 'A');
            else assert(0);
        }

        size_t limbs_written = mini_set_str(this->data, s_copy, l);
        assert(limbs_written <= n);

        delete[] s_copy;
    }

    template<size_t n>
    void bigint<n>::print() const
    {
        for (size_t i = n; i > 0; i--) printf("%016lx", data[i - 1]);
        printf("\n");
    }

    template<size_t n>
    bool bigint<n>::operator==(const bigint<n>& other) const
    {
        return (mini_cmp(this->data, other.data, n) == 0);
    }

    template<size_t n>
    bool bigint<n>::operator!=(const bigint<n>& other) const
    {
        return !(operator==(other));
    }

    template<size_t n>
    void bigint<n>::clear()
    {
        mini_zero(this->data, n);
    }

    template<size_t n>
    bool bigint<n>::is_zero() const
    {
        for (size_t i = 0; i < n; ++i)
        {
            if (this->data[i])
            {
                return false;
            }
        }

        return true;
    }

    template<size_t n>
    size_t bigint<n>::num_bits() const
    {
        for (long i = n-1; i >= 0; --i)
        {
            uint64_t x = this->data[i];
            if (x == 0)
            {
                continue;
            }
            else
            {
                return ((i+1) * MINI_NUMB_BITS) - __builtin_clzl(x);
            }
        }
        return 0;
    }

    template<size_t n>
    unsigned long bigint<n>::as_ulong() const
    {
        return this->data[0];
    }

    template<size_t n>
    bool bigint<n>::test_bit(const std::size_t bitno) const
    {
        if (bitno >= n * MINI_NUMB_BITS)
        {
            return false;
        }
        else
        {
            const std::size_t part = bitno/MINI_NUMB_BITS;
            const std::size_t bit = bitno - (MINI_NUMB_BITS*part);
            const uint64_t one = 1;
            return (this->data[part] & (one<<bit)) != 0;
        }
    }

    template<size_t n>
    bigint<n>& bigint<n>::randomize()
    {
        static_assert(MINI_NUMB_BITS == sizeof(uint64_t) * 8, "Wrong MINI_NUMB_BITS value");
        std::random_device rd;
        constexpr size_t num_random_words = sizeof(uint64_t) * n / sizeof(std::random_device::result_type);
        auto random_words = reinterpret_cast<std::random_device::result_type*>(this->data);
        for (size_t i = 0; i < num_random_words; ++i) random_words[i] = rd();

        return (*this);
    }

} // libff
