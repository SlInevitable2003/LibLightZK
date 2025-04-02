/** @file
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#pragma once
#include <cstdint>
#include "bigint.cuh"

namespace libff {

    template<typename GroupT, size_t m>
    GroupT scalar_mul(const GroupT &base, const bigint<m> &scalar);

} // libff

/** @file
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

namespace libff {

    template<typename GroupT, size_t m>
    GroupT scalar_mul(const GroupT &base, const bigint<m> &scalar)
    {
        GroupT result = GroupT::zero();

        bool found_one = false;
        for (long i = static_cast<long>(scalar.max_bits() - 1); i >= 0; --i)
        {
            if (found_one)
            {
                result = result.dbl();
            }

            if (scalar.test_bit(i))
            {
                found_one = true;
                result = result + base;
            }
        }

        return result;
    }

}