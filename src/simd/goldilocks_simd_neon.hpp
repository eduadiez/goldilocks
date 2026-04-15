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
    // Phase 1e: scalar mul+umulh per lane via __uint128_t. Matches Plonky3's
    // aarch64_neon/packing.rs:295-395 approach. Apple Silicon has 2 integer
    // multiply pipes; out-of-order execution interleaves the two lane muls
    // to hit ~2x integer-mul throughput. NEON vmull_u32 has worse latency and
    // fewer issue slots on M-series than scalar mul+umulh.
    //
    // Reduction follows scalar Goldilocks::mul: 2^64 ≡ 2^32 - 1 (mod p).
    // Non-canonical output in [0, 2^64); public mul() canonicalizes.
    static inline __attribute__((always_inline)) uint64_t mul_scalar(uint64_t a, uint64_t b) {
        __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
        uint64_t c_lo = (uint64_t)prod;
        uint64_t c_hi = (uint64_t)(prod >> 64);
        uint64_t c_hi_lo = (uint32_t)c_hi;
        uint64_t c_hi_hi = c_hi >> 32;

        uint64_t s = c_lo - c_hi;
        bool borrow = s > c_lo;
        uint64_t s2 = s + (c_hi_lo << 32);
        bool carry2 = s2 < s;
        // Plonky3 shift-sub: c_hi_hi * EPSILON = (c_hi_hi << 32) - c_hi_hi
        uint64_t s3 = s2 + ((c_hi_hi << 32) - c_hi_hi);
        bool carry3 = s3 < s2;

        int adj = (int)carry2 + (int)carry3 - (int)borrow;
        uint64_t r = s3 + (uint64_t)adj * (uint64_t)GOLDILOCKS_PRIME_NEG;
        if (adj < 0) r = s3 - (uint64_t)GOLDILOCKS_PRIME_NEG;
        return r;
    }

    static inline __attribute__((always_inline)) Vec mul_reduced(Vec a, Vec b) {
        uint64_t a0 = vgetq_lane_u64(a, 0);
        uint64_t a1 = vgetq_lane_u64(a, 1);
        uint64_t b0 = vgetq_lane_u64(b, 0);
        uint64_t b1 = vgetq_lane_u64(b, 1);
        uint64_t r0 = mul_scalar(a0, b0);
        uint64_t r1 = mul_scalar(a1, b1);
        uint64_t tmp[2] = {r0, r1};
        return vld1q_u64(tmp);
    }

    // Public canonical mul — preserves the [0, P) output contract.
    static inline __attribute__((always_inline)) Vec mul(Vec a, Vec b) {
        uint64x2_t r = mul_reduced(a, b);
        uint64x2_t ge_p_mask = vcgeq_u64(r, P);
        return vsubq_u64(r, vandq_u64(ge_p_mask, P));
    }

    // Non-canonical square. Alias for mul_reduced(a, a).
    static inline __attribute__((always_inline)) Vec square_reduced(Vec a) {
        return mul_reduced(a, a);
    }
    static inline __attribute__((always_inline)) Vec square(Vec a) {
        return mul(a, a);
    }

    // Final canonicalizer for use right before store. Non-canonical → canonical.
    // Must handle inputs up to [0, 2^64). If r > P but r < 2P, one subtract suffices.
    // Wrap-around corrections in add can push intermediate values close to 2^64; the
    // double-subtract ensures r ∈ [0, P) on exit.
    static inline __attribute__((always_inline)) Vec canonicalize(Vec r) {
        uint64x2_t ge_p = vcgeq_u64(r, P);
        r = vsubq_u64(r, vandq_u64(ge_p, P));
        ge_p = vcgeq_u64(r, P);
        return vsubq_u64(r, vandq_u64(ge_p, P));
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
