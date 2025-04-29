#pragma once
#include "alt_bn128_t.cuh"

#include "dmont_t.cuh"
typedef uint64_t vec256[4];

namespace alt_bn128 {

    typedef dmont_t<254, device::ALT_BN128_P, device::ALT_BN128_M0, device::ALT_BN128_RR, device::ALT_BN128_one, device::ALT_BN128_Px4> dfp_mont;
    struct dfp_t : public dfp_mont {
        using mem_t = dfp_t;
        __device__ __forceinline__ dfp_t() {}
        __device__ __forceinline__ dfp_t(const dfp_mont& a) : dfp_mont(a) {}
        template<typename... Ts> constexpr dfp_t(Ts... a)  : dfp_mont{a...} {}
    };
    typedef dmont_t<254, device::ALT_BN128_r, device::ALT_BN128_m0, device::ALT_BN128_rRR, device::ALT_BN128_rone, device::ALT_BN128_rx4> dfr_mont;
    struct dfr_t : public dfr_mont {
        using mem_t = dfr_t;
        __device__ __forceinline__ dfr_t() {}
        __device__ __forceinline__ dfr_t(const dfr_mont& a) : dfr_mont(a) {}
        template<typename... Ts> constexpr dfr_t(Ts... a)  : dfr_mont{a...} {}
    };

} // namespace alt_bn128

#include "jacobian_t.cuh"

namespace alt_bn128 {
    typedef jacobian_t<dfp_t> dg1_jacobian;
    struct dg1_t : public dg1_jacobian {
    private:
        static __device__ __forceinline__ uint32_t laneid() { return threadIdx.x % 32; }
    public:
        __device__ __forceinline__ dg1_t() {}
        __device__ __forceinline__ dg1_t(const dg1_jacobian& a) : dg1_jacobian(a) {}
        template<typename... Ts> constexpr dg1_t(Ts... a)  : dg1_jacobian{a...} {}

        __device__ __forceinline__ void zero() { 
            dg1_jacobian::set_X(dfp_t::one(1));
            dg1_jacobian::set_Y(dfp_t::one());
            dg1_jacobian::set_Z(dfp_t::one(1));
        }
        __device__ __forceinline__ void one() { 
            dg1_jacobian::set_X(dfp_t::one());
            dg1_jacobian::set_Y(dfp_t::one() + dfp_t::one());
            dg1_jacobian::set_Z(dfp_t::one());
        }

        __device__ __forceinline__ void read_from(const void *src) {
            for (size_t i = 0; i < 8; i++) {
                dg1_jacobian::set_X(*((dfp_t*)src + (laneid() % 8)));
                dg1_jacobian::set_Y(*((dfp_t*)src + (laneid() % 8) + 8));
                dg1_jacobian::set_Z(*((dfp_t*)src + (laneid() % 8) + 16));
            }
        }
        __device__ __forceinline__ void write_to(const void *dest) {
            for (size_t i = 0; i < 8; i++) {
                *((dfp_t*)dest + (laneid() % 8)) = dg1_jacobian::out_X();
                *((dfp_t*)dest + (laneid() % 8) + 8) = dg1_jacobian::out_Y();
                *((dfp_t*)dest + (laneid() % 8) + 16) = dg1_jacobian::out_Z();
            }
        }
    };
} // namespace alt_bn128