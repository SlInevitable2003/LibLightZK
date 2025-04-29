#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include "pow.cuh"

#define inline __device__ __forceinline__
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

template <
    const size_t N, 
    const uint32_t MOD[(N + 31) / 32], 
    const uint32_t &M0, 
    const uint32_t RR[(N + 31) / 32], 
    const uint32_t ONE[(N + 31) / 32], 
    const uint32_t MODx[(N + 31) / 32] = MOD
>
class dmont_t
{
public:
    static const size_t nbits = N;

protected:
    static const size_t n = (N + 31) / 32;

private:
    uint32_t even;

    static inline uint32_t laneid() { return threadIdx.x % 32; }

    static inline void cadd_n(uint32_t *acc, const uint32_t *a, uint32_t *carry_out, size_t n = (N + 31) / 32)
    {
        uint32_t carry;
        asm volatile("add.cc.u32 %0, %0, %1;\n": "+r" (*acc) : "r" (*a));
        asm volatile("addc.cc.u32 %0, 0, 0;\n": "=r" (carry));

        asm("{ .reg.pred %sel;");
        for (size_t i = 1; i < n; i++) {
            uint32_t sel = (laneid() % n) == i;
            asm("setp.ne.s32 %sel, %0, 0;" : : "r"(sel));
            uint32_t carry_in = __shfl_up_sync(__activemask(), carry, 1);
            asm("selp.u32 %0, %1, 0, %sel;" : "=r"(carry_in) : "r"(carry_in));
            asm volatile("add.cc.u32 %0, %0, %1;\n": "+r" (*acc): "r" (carry_in));
            asm volatile("addc.cc.u32 %0, %0, 0;\n": "+r" (carry));
        }
        asm("}");
        if (((laneid() % n) == (n-1)) && carry_out) *carry_out = carry;
    }

    inline void final_subc(uint32_t carry)
    {
        uint32_t tmp, borrow;
        asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(tmp) : "r"(even), "r"(MOD[laneid() % n]));
        asm volatile("subc.cc.u32 %0, 0, 0;\n": "=r" (borrow));

        asm("{ .reg.pred %sel;");
        for (size_t i = 1; i < n; i++) {
            uint32_t sel = (laneid() % n) == i;
            asm("setp.ne.s32 %sel, %0, 0;" : : "r"(sel));
            uint32_t borrow_in = __shfl_up_sync(__activemask(), borrow, 1);
            asm("selp.u32 %0, %1, 0, %sel;" : "=r"(borrow_in) : "r"(-borrow_in));
            asm volatile("sub.cc.u32 %0, %0, %1;\n": "+r" (tmp): "r" (borrow_in));
            asm volatile("subc.cc.u32 %0, %0, 0;\n": "+r" (borrow));
        }
        asm("}");
        if ((laneid() % n) == (n-1)) carry -= borrow;
        carry = __shfl_sync(__activemask(), carry, laneid() / n * n + n - 1);

        asm("{ .reg.pred %top;");
        asm("setp.eq.u32 %top, %0, 0;" : :"r"(carry));
        asm("@%top mov.b32 %0, %1;" : "+r"(even) : "r"(tmp));
        asm("}");
    }

    static inline void mul_n(uint32_t *low, uint32_t *high, const uint32_t *a, uint32_t bi, size_t n = (N + 31) / 32)
    {
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(*low), "=r"(*high) : "r"(*a), "r"(bi));
    }

public:
    static inline const dmont_t &one()
    {
        return *reinterpret_cast<const dmont_t *>(ONE + (laneid() % 8));
    }
    static inline dmont_t one(int or_zero)
    {
        dmont_t ret;
        asm("{ .reg.pred %or_zero;");
        asm("setp.ne.s32 %or_zero, %0, 0;" : : "r"(or_zero));
        asm("selp.u32 %0, 0, %1, %or_zero;" : "=r"(ret.even) : "r"(ONE[laneid() % 8]));
        asm("}");
        return ret;
    }

    inline dmont_t &operator+=(const dmont_t &b)
    {
        uint32_t carry;
        cadd_n(&even, &b.even, &carry);
        final_subc(carry);
        return *this;
    }
    friend inline dmont_t operator+(dmont_t a, const dmont_t &b) { return a += b; }

    inline dmont_t &operator-=(const dmont_t &b)
    {
        uint32_t tmp, borrow;

        asm volatile("sub.cc.u32 %0, %0, %1;" : "+r"(even) : "r"(b.even));
        asm volatile("subc.cc.u32 %0, 0, 0;\n": "+r" (borrow));

        asm("{ .reg.pred %sel;");
        for (size_t i = 1; i < n; i++) {
            uint32_t sel = (laneid() % n) == i;
            asm("setp.ne.s32 %sel, %0, 0;" : : "r"(sel));
            uint32_t borrow_in = __shfl_up_sync(__activemask(), borrow, 1);
            asm("selp.u32 %0, %1, 0, %sel;" : "=r"(borrow_in) : "r"(-borrow_in));
            asm volatile("sub.cc.u32 %0, %0, %1;\n": "+r" (even): "r" (borrow_in));
            asm volatile("subc.cc.u32 %0, %0, 0;\n": "+r" (borrow));
        }
        asm("}");

        uint32_t carry;
        asm volatile("add.cc.u32 %0, %1, %2;\n": "=r"(tmp) : "r"(even), "r"(MOD[laneid() % n]));
        asm volatile("addc.cc.u32 %0, 0, 0;\n": "+r" (carry));

        asm("{ .reg.pred %sel;");
        for (size_t i = 1; i < n; i++) {
            uint32_t sel = (laneid() % n) == i;
            asm("setp.ne.s32 %sel, %0, 0;" : : "r"(sel));
            uint32_t carry_in = __shfl_up_sync(__activemask(), carry, 1);
            asm("selp.u32 %0, %1, 0, %sel;" : "=r"(carry_in) : "r"(carry_in));
            asm volatile("add.cc.u32 %0, %0, %1;\n": "+r" (tmp): "r" (carry_in));
            asm volatile("addc.cc.u32 %0, %0, 0;\n": "+r" (carry));
        }
        asm("}");

        borrow = __shfl_sync(__activemask(), borrow, laneid() / n * n + n - 1);
        asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" ::"r"(borrow));
        asm("@%top mov.b32 %0, %1;" : "+r"(even) : "r"(tmp));
        asm("}");

        return *this;
    }
    friend inline dmont_t operator-(dmont_t a, const dmont_t &b) { return a -= b; }

    inline dmont_t &operator*=(const dmont_t &b)
    {
        uint32_t Al = 0, Ah = 0;
        uint32_t low, high;
        uint32_t carry;

        uint32_t b0 = __shfl_sync(__activemask(), b.even, laneid() / n * n);
        for (size_t i = 0; i < n; i++) {
            uint32_t A0 = __shfl_sync(__activemask(), Al, laneid() / n * n);
            uint32_t ai = __shfl_sync(__activemask(), even, laneid() / n * n + i);
            uint32_t ui = (A0 + ai * b0) * M0;
            uint32_t tail = laneid() % n == (n-1);

            mul_n(&low, &high, &b.even, ai);
            cadd_n(&Al, &low, &carry);
            if (tail) Ah += carry;
            cadd_n(&Ah, &high, 0);

            mul_n(&low, &high, MOD + (laneid() % n), ui);
            cadd_n(&Al, &low, &carry);
            if (tail) Ah += carry;
            cadd_n(&Ah, &high, 0);

            high = __shfl_up_sync(__activemask(), Ah, 1);
            if (laneid() % n == 0) high = 0;
            cadd_n(&Al, &high, &carry);
            if (tail) Ah += carry;
            asm("{ .reg.pred %sel;");
            asm("setp.ne.s32 %sel, %0, 0;" : : "r"(tail));
            high = __shfl_down_sync(__activemask(), Al, 1);
            asm("selp.u32 %0, %1, %2, %sel;" : "=r"(Al) : "r"(Ah), "r"(high));
            Ah = 0;
            asm("}");
        }
        even = Al;
        high = __shfl_up_sync(__activemask(), Ah, 1);
        if (laneid() % n == 0) high = 0;
        cadd_n(&even, &high, &carry);
        final_subc(carry);
    }
    friend inline dmont_t operator*(dmont_t a, const dmont_t &b) { return a *= b; }

    inline dmont_t &sqr()
    {
        return *this *= *this;
    }
    inline dmont_t &operator^=(uint32_t p)
    {
        return pow_byref(*this, p);
    }
    friend inline dmont_t operator^(dmont_t a, uint32_t p)
    {
        return a ^= p;
    }
};

#undef inline
#undef asm