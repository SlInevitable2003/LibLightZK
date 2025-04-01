// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace device {
#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64>>32)
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_P[8] = {
        TO_CUDA_T(0x3c208c16d87cfd47), TO_CUDA_T(0x97816a916871ca8d),
        TO_CUDA_T(0xb85045b68181585d), TO_CUDA_T(0x30644e72e131a029)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_RR[8] = { /* (1<<512)%P */
        TO_CUDA_T(0xf32cfc5b538afa89), TO_CUDA_T(0xb5e71911d44501fb),
        TO_CUDA_T(0x47ab1eff0a417ff6), TO_CUDA_T(0x06d89f71cab8351f),
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_one[8] = { /* (1<<256)%P */
        TO_CUDA_T(0xd35d438dc58f0d9d), TO_CUDA_T(0x0a78eb28f5c70b3d),
        TO_CUDA_T(0x666ea36f7879462c), TO_CUDA_T(0x0e0a77c19a07df2f)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_Px4[8] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xf082305b61f3f51c), TO_CUDA_T(0x5e05aa45a1c72a34),
        TO_CUDA_T(0xe14116da06056176), TO_CUDA_T(0xc19139cb84c680a6)
    };
    static __device__ __constant__ const uint32_t ALT_BN128_M0 = 0xe4866389;

    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_r[8] = {
        TO_CUDA_T(0x43e1f593f0000001), TO_CUDA_T(0x2833e84879b97091),
        TO_CUDA_T(0xb85045b68181585d), TO_CUDA_T(0x30644e72e131a029)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rRR[8] = { /* (1<<512)%P */
        TO_CUDA_T(0x1bb8e645ae216da7), TO_CUDA_T(0x53fe3ab1e35c59e3),
        TO_CUDA_T(0x8c49833d53bb8085), TO_CUDA_T(0x0216d0b17f4e44a5)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rone[8] = { /* (1<<256)%P */
        TO_CUDA_T(0xac96341c4ffffffb), TO_CUDA_T(0x36fc76959f60cd29),
        TO_CUDA_T(0x666ea36f7879462e), TO_CUDA_T(0x0e0a77c19a07df2f)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rx4[8] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0x0f87d64fc0000004), TO_CUDA_T(0xa0cfa121e6e5c245),
        TO_CUDA_T(0xe14116da06056174), TO_CUDA_T(0xc19139cb84c680a6)
    };
    static __device__ __constant__ const uint32_t ALT_BN128_m0 = 0xefffffff;
}

#include "mont_t.cuh"
typedef uint64_t vec256[4];

namespace alt_bn128 {

    typedef mont_t<254, device::ALT_BN128_P, device::ALT_BN128_M0, device::ALT_BN128_RR, device::ALT_BN128_one, device::ALT_BN128_Px4> fp_mont;
    struct fp_t : public fp_mont {
        using mem_t = fp_t;
        __device__ __forceinline__ fp_t() {}
        __device__ __forceinline__ fp_t(const fp_mont& a) : fp_mont(a) {}
        template<typename... Ts> constexpr fp_t(Ts... a)  : fp_mont{a...} {}
    };
    typedef mont_t<254, device::ALT_BN128_r, device::ALT_BN128_m0, device::ALT_BN128_rRR, device::ALT_BN128_rone, device::ALT_BN128_rx4> fr_mont;
    struct fr_t : public fr_mont {
        using mem_t = fr_t;
        __device__ __forceinline__ fr_t() {}
        __device__ __forceinline__ fr_t(const fr_mont& a) : fr_mont(a) {}
        template<typename... Ts> constexpr fr_t(Ts... a)  : fr_mont{a...} {}
    };

} // namespace alt_bn128

#include "jacobian_t.cuh"

namespace alt_bn128 {
    typedef jacobian_t<fp_t> g1_jacobian;
    struct g1_t : public g1_jacobian {
        __device__ __forceinline__ g1_t() {}
        __device__ __forceinline__ g1_t(const g1_jacobian& a) : g1_jacobian(a) {}
        template<typename... Ts> constexpr g1_t(Ts... a)  : g1_jacobian{a...} {}

        __device__ __forceinline__ void zero() { ((fp_t*)this)[0] = ((fp_t*)this)[2] = fp_t::one(1), ((fp_t*)this)[1] = fp_t::one(); }
        __device__ __forceinline__ void one() { ((fp_t*)this)[0] = ((fp_t*)this)[2] = fp_t::one(), ((fp_t*)this)[1] = fp_t::one() << 1; }
    };
} // namespace alt_bn128

# ifndef WARP_SZ
#  define WARP_SZ 32
# endif

# define inline __device__ __forceinline__
# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

namespace alt_bn128 {

    class fp2_t : public fp_mont {
    private:
        static inline uint32_t laneid()
        {   return threadIdx.x % WARP_SZ;   }

    public:
        static const uint32_t degree = 2;

        class mem_t { friend fp2_t;
            fp_mont x[2];

        public:
            inline operator fp2_t() const           { return x[threadIdx.x&1]; }
            inline void zero()                      { x[threadIdx.x&1].zero(); }
            inline void to()                        { x[threadIdx.x&1].to();   }
            inline void from()                      { x[threadIdx.x&1].from(); }
            inline mem_t& operator=(const fp2_t& a)
            {   x[threadIdx.x&1] = a; return *this;   }
        };

        inline fp2_t()                              {}
        inline fp2_t(const fp_mont& a) : fp_mont(a) {}
        inline fp2_t(const mem_t* p)                { *this = p->x[threadIdx.x&1]; }
        inline void store(mem_t* p) const           { p->x[threadIdx.x&1] = *this; }

        friend inline fp2_t operator*(const fp2_t& a, const fp2_t& b)
        {
            auto id = laneid();
            auto mask = __activemask();
            auto t0 = b.shfl(id&~1, mask);
            auto t1 = a.shfl(id^1, mask);
            auto t2 = b.shfl(id|1, mask);
            t1.cneg((id&1) == 0);


            return fp_mont(a) * fp_mont(t0) + fp_mont(t1) * fp_mont(t2);
        }
        inline fp2_t& operator*=(const fp2_t& a)
        {   return *this = *this * a;   }

        inline fp2_t& sqr()
        {
            auto id = laneid();
            fp_mont t0 = shfl(id^1, __activemask());
            fp_mont t1 = *this;

            if ((id&1) == 0) {
                t1 = (fp_mont)*this + t0;
                t0 = (fp_mont)*this - t0;
            }
            t0 *= t1;
            t1 = t0 << 1;

            return *this = fp_mont::csel(t1, t0, id&1);
        }
        inline fp2_t& operator^=(int p)
        {   if (p != 2) asm("trap;"); return sqr();     }
        friend inline fp2_t operator^(fp2_t a, int p)
        {   if (p != 2) asm("trap;"); return a.sqr();   }

        friend inline fp2_t operator+(const fp2_t& a, const fp2_t& b)
        {   return (fp_mont)a + (fp_mont)b;   }
        inline fp2_t& operator+=(const fp2_t& b)
        {   return *this = *this + b;   }

        friend inline fp2_t operator-(const fp2_t& a, const fp2_t& b)
        {   return (fp_mont)a - (fp_mont)b;   }
        inline fp2_t& operator-=(const fp2_t& b)
        {   return *this = *this - b;   }

        friend inline fp2_t operator<<(const fp2_t& a, unsigned l)
        {   return (fp_mont)a << l;   }
        inline fp2_t& operator<<=(unsigned l)
        {   return *this = *this << l;   }

        inline fp2_t& cneg(bool flag)
        {   fp_mont::cneg(flag); return *this;  }
        friend inline fp2_t cneg(fp2_t a, bool flag)
        {   return a.cneg(flag);   }

        friend inline fp2_t czero(const fp2_t& a, int set_z)
        {   return czero((fp_mont)a, set_z);   }

        inline bool is_zero() const
        {
            auto ret = __ballot_sync(__activemask(), fp_mont::is_zero());
            return ((ret >> (laneid()&~1)) & 3) == 3;
        }

        inline bool is_zero(const fp2_t& a) const
        {
            auto ret = __ballot_sync(__activemask(), fp_mont::is_zero(a));
            return ((ret >> (laneid()&~1)) & 3) == 3;
        }

        static inline fp2_t one(int or_zero = 0)
        {   return fp_mont::one((laneid()&1) | or_zero);   }

        inline bool is_one() const
        {
            auto id = laneid();
            auto even = ~(0 - (id&1));
            uint32_t is_zero = ((fp_mont)*this)[0] ^ (fp_mont::one()[0] & even);

            for (size_t i = 1; i < n; i++)
                is_zero |= ((fp_mont)*this)[i] ^ (fp_mont::one()[i] & even);

            is_zero = __ballot_sync(__activemask(), is_zero == 0);
            return ((is_zero >> (id&~1)) & 3) == 3;
        }

        inline fp2_t reciprocal() const
        {
            auto a = (fp_mont)*this^2;
            auto b = shfl_xor(a);
            a += b;
            a = ct_inverse_mod_x(a);    // 1/(x[0]^2 + x[1]^2)
            a *= (fp_mont)*this;
            a.cneg(threadIdx.x&1);
            return a;
        }
        friend inline fp2_t operator/(int one, const fp2_t& a)
        {   if (one != 1) asm("trap;"); return a.reciprocal();   }
        friend inline fp2_t operator/(const fp2_t& a, const fp2_t& b)
        {   return a * b.reciprocal();   }
        inline fp2_t& operator/=(const fp2_t& a)
        {   return *this *= a.reciprocal();   }
    };

} // namespace alt_bn128

# undef inline
# undef asm

namespace alt_bn128 {
    typedef jacobian_t<fp2_t> g2_jacobian;
    struct g2_t : public g2_jacobian {
        __device__ __forceinline__ g2_t() {}
        __device__ __forceinline__ g2_t(const g2_jacobian& a) : g2_jacobian(a) {}
        template<typename... Ts> constexpr g2_t(Ts... a)  : g2_jacobian{a...} {}

        __device__ __forceinline__ void zero() { 
            ((fp_t*)this)[0] = ((fp_t*)this)[2] = fp_t::one(1);
            ((fp_t*)this)[1] = fp_mont::csel(fp_t::one(), fp_t::one(1), (threadIdx.x & 1) == 0); 
        }
        __device__ __forceinline__ void one() { 
            ((fp_t*)this)[0] = ((fp_t*)this)[2] = fp_mont::csel(fp_t::one(), fp_t::one(1), (threadIdx.x & 1) == 0);
            ((fp_t*)this)[1] = fp_mont::csel(fp_t::one() << 1, fp_t::one(1), (threadIdx.x & 1) == 0); 
        }

        __device__ __forceinline__ void read_from(const fp_t *src) {
            ((fp_t*)this)[0] = src[0 + (threadIdx.x & 1)];
            ((fp_t*)this)[1] = src[2 + (threadIdx.x & 1)];
            ((fp_t*)this)[2] = src[4 + (threadIdx.x & 1)];
        }
        __device__ __forceinline__ void write_to(void *dest) {
            ((fp_t*)dest)[0 + (threadIdx.x & 1)] = ((fp_t*)this)[0];
            ((fp_t*)dest)[2 + (threadIdx.x & 1)] = ((fp_t*)this)[1];
            ((fp_t*)dest)[4 + (threadIdx.x & 1)] = ((fp_t*)this)[2];
        }
    };
} // namespace alt_bn128
