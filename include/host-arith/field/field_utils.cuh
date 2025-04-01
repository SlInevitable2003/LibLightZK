/** @file
 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#pragma once
#include <cstdint>
#include <vector>
#include "bigint.cuh"

namespace libff {

    template<typename FieldT>
    void batch_invert(std::vector<FieldT> &vec)
    {
        std::vector<FieldT> prod;
        prod.reserve(vec.size());

        FieldT acc = FieldT::one();

        for (auto el : vec)
        {
            assert(!el.is_zero());
            prod.emplace_back(acc);
            acc = acc * el;
        }

        FieldT acc_inverse = acc.inverse();

        for (long i = static_cast<long>(vec.size()-1); i >= 0; --i)
        {
            const FieldT old_el = vec[i];
            vec[i] = acc_inverse * prod[i];
            acc_inverse = acc_inverse * old_el;
        }
    }

} // libff

