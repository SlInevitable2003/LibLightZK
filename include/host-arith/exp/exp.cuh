/** @file
 *****************************************************************************

 Declaration of interfaces for (square-and-multiply) exponentiation.

 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#pragma once

#include <cstdint>
#include "bigint.cuh"

namespace libff {

    template<typename FieldT, size_t m>
    FieldT power(const FieldT &base, const bigint<m> &exponent);

    template<typename FieldT>
    FieldT power(const FieldT &base, const unsigned long exponent);

} // libff

/** @file
 *****************************************************************************

 Implementation of interfaces for (square-and-multiply) exponentiation.

 See exponentiation.hpp .

 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

namespace libff {

template<typename FieldT, size_t m>
FieldT power(const FieldT &base, const bigint<m> &exponent)
{
    FieldT result = FieldT::one();

    bool found_one = false;

    for (long i = exponent.max_bits() - 1; i >= 0; --i)
    {
        if (found_one)
        {
            result = result * result;
        }

        if (exponent.test_bit(i))
        {
            found_one = true;
            result = result * base;
        }
    }

    return result;
}

template<typename FieldT>
FieldT power(const FieldT &base, const unsigned long exponent)
{
    return power<FieldT>(base, bigint<1>(exponent));
}

} // libff