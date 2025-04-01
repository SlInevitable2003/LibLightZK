#pragma once
#include <cstdint>
#include "xpu-vector.cuh"

namespace xpu {

    size_t next_pow2(size_t x) {
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    }

    enum class application_scence { MatMul };
    enum class matrix_type { A, B, C };

    template <typename FieldT>
    struct spmat_descript {
        FieldT *val;
        size_t *col_idx, *row_ptr;
        size_t row, col, cnt;
    };

    template <typename FieldT>
    class spmat {
        vector<FieldT> val;
        vector<size_t> col_idx, row_ptr;
        size_t row, col, cnt;

        void spmat_gen_MatMul(matrix_type mt, size_t p)
        {
            size_t p2 = p * p, p3 = p * p * p;
            size_t M = next_pow2(p3 + p2);
            row = 0, col = 1 + 3 * p2 + p3, cnt = 0;

            val.allocate(((mt == matrix_type::A) ? (2 * p3) : (p3 + p2)), mem_policy::cross_platform);
            col_idx.allocate(((mt == matrix_type::A) ? (2 * p3) : (p3 + p2)), mem_policy::cross_platform);
            row_ptr.allocate(M, mem_policy::cross_platform);

            for (size_t i = 0; i < p3; i++) {
                size_t i1 =  i / p2, i2 = (i % p2) / p, i3 = i % p;
                size_t offset = (mt == matrix_type::A) ? (1 + i2 * p + i1) : ((mt == matrix_type::B) ? (1 + p2 + i1 * p + i3) : (1 + 3 * p2 + i));
                val[cnt] = FieldT::one(), col_idx[cnt] = offset, row_ptr[row++] = cnt++;
            }

            for (size_t i = 0; i < p2; i++) {
                if (mt == matrix_type::A) {
                    row_ptr[row++] = cnt;
                    for (size_t k = 0; k < p; k++) {
                        size_t offset = 1 + 3 * p2 + k * p2 + i;
                        val[cnt] = FieldT::one(), col_idx[cnt] = offset, cnt++;
                    }
                } else {
                    size_t offset = (mt == matrix_type::B) ? 0 : (1 + 2 * p2 + i);
                    val[cnt] = FieldT::one(), col_idx[cnt] = offset, row_ptr[row++] = cnt++;
                }
            }

            for (size_t i = p3 + p2; i < M; i++) row_ptr[row++] = cnt;

            val.store(), col_idx.store(), row_ptr.store();
        }
    public:

        // void print() 
        // {
        //     int *buffer = new int[row * col];
        //     for (size_t i = 0; i < row; i++) {
        //         for (size_t j = 0; j < col; j++) buffer[i * col + j] = 0;
        //     }

        //     for (size_t i = 0; i < row - 1; i++) for (size_t j = row_ptr[i]; j < cnt && j < row_ptr[i + 1]; j++) buffer[i * col + col_idx[j]] = 1;
        //     for (size_t j = row_ptr[row - 1]; j < cnt; j++) buffer[(row - 1) * col + col_idx[j]] = 1;
        //     for (size_t i = 0; i < row; i++) {
        //         for (size_t j = 0; j < col; j++) printf("%d ", buffer[i * col + j]);
        //         printf("\n");
        //     }
        //     printf("\n");

        //     delete[] buffer;
        // }

        spmat() {}

        spmat(application_scence as, matrix_type mt, size_t param1) 
        {
            switch (as) {
                case application_scence::MatMul: { spmat_gen_MatMul(mt, param1); } break;
            }
        }

        void init(application_scence as, matrix_type mt, size_t param1)
        {
            switch (as) {
                case application_scence::MatMul: { spmat_gen_MatMul(mt, param1); } break;
            }
        }

        template <typename devFdT> spmat_descript<devFdT> descript() { return {(devFdT*)val.p(), (size_t*)col_idx.p(), (size_t*)row_ptr.p(), row, col, cnt}; }

        ~spmat() {}
    };

    

}