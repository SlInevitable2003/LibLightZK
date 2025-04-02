#pragma once
#include <cstdint>

int mini_cmp(const uint64_t *src1, const uint64_t *src2, size_t l);
void mini_zero(uint64_t *dest, size_t l);
void mini_copyi(uint64_t *dest, const uint64_t *src, size_t l);
size_t mini_set_str(uint64_t *dest, const unsigned char *str, size_t l);

uint64_t mini_add_1(uint64_t *rp, const uint64_t *up, size_t n, uint64_t vl);
uint64_t mini_sub_1(uint64_t *rp, const uint64_t *up, size_t n, uint64_t vl);

uint64_t mini_add_n(uint64_t *rp, const uint64_t *up, const uint64_t *vp, size_t n);
uint64_t mini_sub_n(uint64_t *rp, const uint64_t *up, const uint64_t *vp, size_t n);

uint64_t mini_sub(uint64_t *rp, const uint64_t *up, size_t un, const uint64_t *vp, size_t vn);

uint64_t mini_addmul_1(uint64_t *rp, const uint64_t *up, size_t n, uint64_t vl);
void mini_mul_n(uint64_t *rp, const uint64_t *up, const uint64_t *vp, size_t n);