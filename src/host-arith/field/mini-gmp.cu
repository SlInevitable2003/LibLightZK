#include "mini-gmp.cuh"
#include "host-def.cuh"

#if defined(__x86_64__) && defined(__GNUC__)
#define umul_ppmm(ph, pl, a, b) \
    do { \
        uint64_t __a = (a), __b = (b); \
        asm ("mulq %3" \
            : "=a" (pl), "=d" (ph) \
            : "%0" (__a), "rm" (__b) \
            : "cc"); \
    } while (0)
#endif

#define add_ssaaaa(sh, sl, ah, al, bh, bl) \
    do { \
        uint64_t __al = (al); \
        uint64_t __bl = (bl); \
        uint64_t __sum = __al + __bl; \
        uint64_t __carry = __sum < __al; \
        sl = __sum; \
        sh = (ah) + (bh) + __carry; \
    } while (0)

int mini_cmp(const uint64_t *src1, const uint64_t *src2, size_t l)
{
    for (size_t i = l; i > 0; i--) if (src1[i - 1] != src2[i - 1]) return (src1[i - 1] < src2[i - 1]) ? (-1) : 1;
    return 0;
}

void mini_zero(uint64_t *dest, size_t l)
{
    memset(dest, 0, l * sizeof(uint64_t));
}

void mini_copyi(uint64_t *dest, const uint64_t *src, size_t l)
{
    memcpy(dest, src, l * sizeof(uint64_t));
}

size_t mini_set_str(uint64_t *dest, const unsigned char *str, size_t l)
{
    size_t ret = 0;
    for (size_t i = 0; i < l; i++) {
        size_t str_idx = l - i - 1;
        size_t limb_idx = i / (MINI_NUMB_BITS / 4), limb_off = i % (MINI_NUMB_BITS / 4);
        ret = max(ret, limb_idx + 1);
        dest[limb_idx] |= uint64_t(str[str_idx]) << (limb_off * 4);
    }
    return ret;
}

uint64_t mini_add_1(uint64_t *rp, const uint64_t *up, size_t n, uint64_t vl)
{
    uint64_t carry = vl;
    size_t i = 0;
    
    while (carry != 0 && i < n) {
        uint64_t ul = up[i];
        uint64_t sum = ul + carry;
        carry = sum < ul;
        rp[i] = sum;
        i++;
    }
    
    if (rp != up) {
        for (; i < n; i++) {
            rp[i] = up[i];
        }
    }
    
    return carry;
}

uint64_t mini_sub_1(uint64_t *rp, const uint64_t *up, size_t n, uint64_t vl)
{
    uint64_t borrow = vl;
    size_t i = 0;
    
    while (borrow != 0 && i < n) {
        uint64_t ul = up[i];
        uint64_t diff = ul - borrow;
        borrow = diff > ul;
        rp[i] = diff;
        i++;
    }
    
    if (rp != up) {
        for (; i < n; i++) {
            rp[i] = up[i];
        }
    }
    
    return borrow;
}

uint64_t mini_add_n(uint64_t *rp, const uint64_t *up, const uint64_t *vp, size_t n)
{
    uint64_t carry = 0;
    size_t i;
    
    for (i = 0; i < n; i++) {
        uint64_t ul = up[i];
        uint64_t vl = vp[i];
        
        uint64_t sum = ul + carry;
        carry = sum < ul;
        
        sum += vl;
        carry += sum < vl;
        
        rp[i] = sum;
    }
    
    return carry;
}

uint64_t mini_sub_n(uint64_t *rp, const uint64_t *up, const uint64_t *vp, size_t n)
{
    uint64_t borrow = 0;
    size_t i;
    
    for (i = 0; i < n; i++) {
        uint64_t ul = up[i];
        uint64_t vl = vp[i];
        
        uint64_t diff = ul - vl - borrow;
        
        borrow = (ul < vl) || (borrow && (ul == vl));
        
        rp[i] = diff;
    }
    
    return borrow;
}

uint64_t mini_sub(uint64_t *rp, const uint64_t *up, size_t un, const uint64_t *vp, size_t vn)
{
    if (vn == 0) {
        if (rp != up) {
            for (size_t i = 0; i < un; i++) {
                rp[i] = up[i];
            }
        }
        return 0;
    }

    if (vn == 1) return mini_sub_1(rp, up, un, vp[0]);
    if (un == vn) return mini_sub_n(rp, up, vp, un);

    uint64_t borrow;
    borrow = mini_sub_n(rp, up, vp, vn);
    if (un > vn) borrow = mini_sub_1(rp + vn, up + vn, un - vn, borrow);
    return borrow;
}

uint64_t mini_addmul_1(uint64_t *rp, const uint64_t *up, size_t n, uint64_t vl)
{
    uint64_t cy_limb = 0;
    size_t i;
    
    for (i = 0; i < n; i++) {
        uint64_t ul = up[i];
        uint64_t rl = rp[i];
        uint64_t prod_high, prod_low;

        umul_ppmm(prod_high, prod_low, ul, vl);
        add_ssaaaa(prod_high, prod_low, prod_high, prod_low, 0, cy_limb);
        add_ssaaaa(cy_limb, rp[i], prod_high, prod_low, 0, rl);
    }
    
    return cy_limb;
}

void mini_mul_n(uint64_t *rp, const uint64_t *up, const uint64_t *vp, size_t n)
{
    size_t i, j;
    
    for (i = 0; i < 2 * n; i++) rp[i] = 0;
    
    for (i = 0; i < n; i++) {
        uint64_t v_limb = vp[i];
        if (v_limb != 0) {
            uint64_t cy_limb = 0;
            
            for (j = 0; j < n; j++) {
                uint64_t u_limb = up[j];
                uint64_t prod_low, prod_high;
                
                umul_ppmm(prod_high, prod_low, u_limb, v_limb);
                
                prod_low += cy_limb;
                prod_high += (prod_low < cy_limb);
                
                prod_low += rp[i + j];
                prod_high += (prod_low < rp[i + j]);
                
                rp[i + j] = prod_low;
                cy_limb = prod_high;
            }
            rp[i + n] = cy_limb;
        }
    }
}