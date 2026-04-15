#ifndef GOLDILOCKS_SIMD_NEON_HPP
#define GOLDILOCKS_SIMD_NEON_HPP

#include "../platform.hpp"
#ifndef GOLDILOCKS_HAS_NEON
#  error "goldilocks_simd_neon.hpp included without GOLDILOCKS_HAS_NEON"
#endif

#include <arm_neon.h>
#include <cstdint>
#include "../goldilocks_base_field.hpp"
#include "goldilocks_simd_backends.hpp"

namespace goldilocks {
namespace simd {

template <> struct GLSimd<Neon> {
    using Element = Goldilocks::Element;
    using Vec     = uint64x2_t;
    static constexpr size_t Lanes = 2;

    // Constants (broadcast across both lanes)
    static inline const Vec P      = vdupq_n_u64(GOLDILOCKS_PRIME);
    static inline const Vec P_n    = vdupq_n_u64(GOLDILOCKS_PRIME_NEG);
    static inline const Vec MSB    = vdupq_n_u64(MSB_);
    static inline const Vec P_s    = veorq_u64(vdupq_n_u64(GOLDILOCKS_PRIME), vdupq_n_u64(MSB_));
    static inline const Vec sqmask = vdupq_n_u64(0x1FFFFFFFFULL);

    // Memory
    static inline __attribute__((always_inline)) Vec load(const Element* p) {
        return vld1q_u64(reinterpret_cast<const uint64_t*>(p));
    }
    static inline __attribute__((always_inline)) Vec load_aligned(const Element* p) {
        return vld1q_u64(reinterpret_cast<const uint64_t*>(__builtin_assume_aligned(p, 16)));
    }
    static inline __attribute__((always_inline)) void store(Element* p, Vec a) {
        vst1q_u64(reinterpret_cast<uint64_t*>(p), a);
    }
    static inline __attribute__((always_inline)) void store_aligned(Element* p, Vec a) {
        vst1q_u64(reinterpret_cast<uint64_t*>(__builtin_assume_aligned(p, 16)), a);
    }

    // Construction
    // Lane-order convention: set(a0, a1) puts a0 in lane 0 (low).
    static inline __attribute__((always_inline)) Vec set(uint64_t a0, uint64_t a1) {
        uint64_t tmp[2] = {a0, a1};
        return vld1q_u64(tmp);
    }
    static inline __attribute__((always_inline)) Vec splat(uint64_t x) { return vdupq_n_u64(x); }
    static inline __attribute__((always_inline)) Vec copy(Vec a)       { return a; }

    // Bit / comparison primitives (declared before arithmetic for inline usage)
    static inline __attribute__((always_inline)) Vec shift(Vec a) { return veorq_u64(a, MSB); }
    static inline __attribute__((always_inline)) Vec cmpgt_biased(Vec a, Vec b) {
        // Signed compare on already-biased (shifted) values gives unsigned cmp.
        return vreinterpretq_u64_s64(
                    vcgtq_s64(vreinterpretq_s64_u64(a),
                              vreinterpretq_s64_u64(b)));
    }
    static inline __attribute__((always_inline)) Vec toCanonical_s(Vec a_s) {
        // If a_s < P_s (signed), then a < P (unsigned), leave it; else add P_n (2^32-1).
        Vec mask = cmpgt_biased(P_s, a_s);        // a_s < P_s ? all-ones : 0
        Vec corr = vbicq_u64(P_n, mask);           // P_n AND NOT mask -> 0 if mask=1, P_n if mask=0
        return vaddq_u64(a_s, corr);
    }
    static inline __attribute__((always_inline)) Vec blend(Vec a, Vec b, Vec mask) {
        return vbslq_u64(mask, b, a);
    }
    static inline __attribute__((always_inline)) Vec and_(Vec a, Vec b) {
        return vandq_u64(a, b);
    }

    // Phase 1d: simplified add/sub using direct unsigned comparison (vcgtq_u64,
    // available in ARMv8). Saves ~3 NEON ops per add vs the shifted-canonical
    // trick that AVX2 needs for its signed-only 64-bit comparison.
    // Matches scalar Goldilocks::add bit-for-bit (both produce non-canonical
    // output in [0, 2^64)).
    static inline __attribute__((always_inline)) Vec add(Vec a, Vec b) {
        Vec r = vaddq_u64(a, b);
        Vec wrap1 = vcgtq_u64(a, r);               // r wrapped iff r < a
        Vec corr1 = vandq_u64(wrap1, P_n);
        r = vaddq_u64(r, corr1);
        Vec wrap2 = vcgtq_u64(corr1, r);           // rare second wrap
        Vec corr2 = vandq_u64(wrap2, P_n);
        return vaddq_u64(r, corr2);
    }
    static inline __attribute__((always_inline)) Vec sub(Vec a, Vec b) {
        Vec r = vsubq_u64(a, b);
        Vec borrow1 = vcgtq_u64(r, a);             // borrow iff r > a unsigned
        Vec corr1 = vandq_u64(borrow1, P_n);
        Vec prev = r;
        r = vsubq_u64(r, corr1);
        Vec borrow2 = vcgtq_u64(r, prev);          // rare second underflow
        Vec corr2 = vandq_u64(borrow2, P_n);
        return vsubq_u64(r, corr2);
    }
    static inline __attribute__((always_inline)) Vec add_small(Vec a, Vec b) {
        // b assumed in [0, 2^32): c0_s cannot wrap (a_sc < 2^64 - 2^32).
        Vec a_s  = shift(a);
        Vec a_sc = toCanonical_s(a_s);
        Vec c0_s = vaddq_u64(a_sc, b);
        return shift(c0_s);
    }

    // ------------------------------------------------------------------------
    // NEON-native 64x64 → 128 multiply + Goldilocks reduction.
    //
    // For each lane k ∈ {0, 1}:
    //   Given a, b ∈ [0, 2^64),
    //   (1) Compute the 128-bit product c_hi:c_lo = a * b using 4× vmull_u32:
    //          a = a_hi * 2^32 + a_lo,  b = b_hi * 2^32 + b_lo
    //          ll = a_lo * b_lo   (fits 64 bits)
    //          lh = a_lo * b_hi   (fits 64 bits)
    //          hl = a_hi * b_lo   (fits 64 bits)
    //          hh = a_hi * b_hi   (fits 64 bits)
    //          mid     = lh + hl                              (may carry)
    //          c_lo    = ll + (mid << 32)                     (may carry)
    //          c_hi    = hh + (mid >> 32) + carry(mid) * 2^32
    //                       + carry(c_lo)
    //
    //   (2) Reduce using 2^64 ≡ 2^32 - 1 (mod p):
    //          Let c_hi_lo = c_hi & (2^32 - 1), c_hi_hi = c_hi >> 32.
    //          s  = c_lo - c_hi                               (borrow b1)
    //          s2 = s + (c_hi_lo << 32)                       (carry c2)
    //          s3 = s2 + c_hi_hi * CQ          where CQ=2^32-1 (carry c3)
    //          adj = c2 + c3 - b1   ∈ {-1, 0, 1, 2}
    //          r  = s3 + adj * CQ
    //          if (r >= p) r -= p
    //
    //   This matches the existing scalar Goldilocks::mul algorithm bit-for-bit.
    //   No __uint128_t is used. No movehdup-style 32-bit lane extraction tricks.
    // ------------------------------------------------------------------------
    static inline __attribute__((always_inline)) Vec mul(Vec a, Vec b) {
        // --- Step 1: 4× vmull_u32 → c_hi:c_lo ---
        uint32x2_t a_lo32 = vmovn_u64(a);
        uint32x2_t a_hi32 = vshrn_n_u64(a, 32);
        uint32x2_t b_lo32 = vmovn_u64(b);
        uint32x2_t b_hi32 = vshrn_n_u64(b, 32);

        uint64x2_t ll = vmull_u32(a_lo32, b_lo32);
        uint64x2_t lh = vmull_u32(a_lo32, b_hi32);
        uint64x2_t hl = vmull_u32(a_hi32, b_lo32);
        uint64x2_t hh = vmull_u32(a_hi32, b_hi32);

        // mid = lh + hl (may overflow 64 bits)
        uint64x2_t mid = vaddq_u64(lh, hl);
        uint64x2_t mid_carry = vcltq_u64(mid, lh);       // all-ones if carry
        uint64x2_t mid_carry_bit = vandq_u64(mid_carry, vdupq_n_u64(1));

        // c_lo = ll + (mid << 32)
        uint64x2_t mid_shl32 = vshlq_n_u64(mid, 32);
        uint64x2_t c_lo = vaddq_u64(ll, mid_shl32);
        uint64x2_t c_lo_carry = vcltq_u64(c_lo, ll);
        uint64x2_t c_lo_carry_bit = vandq_u64(c_lo_carry, vdupq_n_u64(1));

        // c_hi = hh + (mid >> 32) + (mid_carry_bit << 32) + c_lo_carry_bit
        uint64x2_t mid_shr32 = vshrq_n_u64(mid, 32);
        uint64x2_t c_hi = vaddq_u64(hh, mid_shr32);
        c_hi = vaddq_u64(c_hi, vshlq_n_u64(mid_carry_bit, 32));
        c_hi = vaddq_u64(c_hi, c_lo_carry_bit);

        // --- Step 2: Goldilocks reduction ---
        const uint64x2_t mask32 = vdupq_n_u64(0xFFFFFFFFULL);
        uint64x2_t c_hi_lo = vandq_u64(c_hi, mask32);
        uint64x2_t c_hi_hi = vshrq_n_u64(c_hi, 32);

        // s = c_lo - c_hi, borrow if s > c_lo
        uint64x2_t s = vsubq_u64(c_lo, c_hi);
        uint64x2_t borrow_mask = vcgtq_u64(s, c_lo);
        uint64x2_t borrow_bit = vandq_u64(borrow_mask, vdupq_n_u64(1));

        // s2 = s + (c_hi_lo << 32), carry if s2 < s
        uint64x2_t s2 = vaddq_u64(s, vshlq_n_u64(c_hi_lo, 32));
        uint64x2_t carry2_mask = vcltq_u64(s2, s);
        uint64x2_t carry2_bit = vandq_u64(carry2_mask, vdupq_n_u64(1));

        // s3 = s2 + c_hi_hi * CQ  (c_hi_hi < 2^32, CQ = 2^32-1 < 2^32, product < 2^64)
        uint32x2_t c_hi_hi_32 = vmovn_u64(c_hi_hi);
        uint32x2_t cq_32 = vdup_n_u32((uint32_t)GOLDILOCKS_PRIME_NEG);
        uint64x2_t hh_times_cq = vmull_u32(c_hi_hi_32, cq_32);
        uint64x2_t s3 = vaddq_u64(s2, hh_times_cq);
        uint64x2_t carry3_mask = vcltq_u64(s3, s2);
        uint64x2_t carry3_bit = vandq_u64(carry3_mask, vdupq_n_u64(1));

        // r = s3 + (carry2 + carry3) * CQ - borrow * CQ
        // Use unsigned arithmetic because CQ = 2^32 - 1 does NOT fit in int32 (it sign-extends to -1).
        uint64x2_t pos_sum = vaddq_u64(carry2_bit, carry3_bit);      // 0, 1, or 2 per lane
        uint32x2_t pos_sum32 = vmovn_u64(pos_sum);
        uint32x2_t cq_u32 = vdup_n_u32((uint32_t)GOLDILOCKS_PRIME_NEG);
        uint64x2_t pos_cq = vmull_u32(pos_sum32, cq_u32);            // (0..2) * CQ, fits in 33 bits

        uint64x2_t cq_vec = vdupq_n_u64((uint64_t)GOLDILOCKS_PRIME_NEG);
        uint64x2_t neg_cq = vandq_u64(borrow_mask, cq_vec);          // CQ if borrow, else 0

        uint64x2_t r = vaddq_u64(s3, pos_cq);
        r = vsubq_u64(r, neg_cq);

        // Final canonicalize: if r >= p then r -= p
        uint64x2_t ge_p_mask = vcgeq_u64(r, P);
        uint64x2_t sub_amt = vandq_u64(ge_p_mask, P);
        return vsubq_u64(r, sub_amt);
    }

    static inline __attribute__((always_inline)) Vec square(Vec a) {
        return mul(a, a);  // Phase 1: correctness-first. Specialize later if profiling demands.
    }

    // Composite / lane mixing
    static inline __attribute__((always_inline)) Vec permute_lanes(Vec a, Vec b) {
        return vextq_u64(a, b, 1);
    }
    static inline __attribute__((always_inline)) Vec mask_lane0_zero(Vec v) {
        // Zero lane 0, keep lane 1. AVX2 needs a 4-lane mask; NEON uses blend with {0, all_ones}.
        uint64_t mask[2] = {0ULL, ~0ULL};
        Vec m = vld1q_u64(mask);
        return vandq_u64(v, m);
    }
};

} // namespace simd
} // namespace goldilocks

#endif // GOLDILOCKS_SIMD_NEON_HPP
