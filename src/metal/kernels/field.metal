// field.metal — Goldilocks field arithmetic for Apple Metal GPU
//
// Prime:  p = 0xFFFFFFFF00000001  (2^64 - 2^32 + 1)
// CQ:     0xFFFFFFFF             (= 2^32 - 1 = -p mod 2^64 + 1 = epsilon)
//
// Lazy-reduce contract (mirrors NEON goldilocks_simd_neon.hpp:143-161):
//   - gl_mul returns in [0, 2p), NOT [0, p).
//   - gl_add and gl_sub return fully reduced in [0, p).
//   - gl_canonicalize brings [0, 2p) -> [0, p); call only at kernel exit.
//
// Metal 3 / Apple Silicon: ulong is 64-bit; metal::mulhi(ulong, ulong) returns
// the high 64 bits of the 128-bit product.

#pragma once
#include <metal_stdlib>
using namespace metal;

constant ulong GL_PRIME = 0xFFFFFFFF00000001UL;
constant ulong GL_CQ    = 0xFFFFFFFFUL;  // 2^32 - 1

// gl_add: fully reduced result in [0, p)
// Algorithm: s = a + b; on 64-bit overflow add CQ (= 2^32-1);
// if that also wraps add CQ again. Two corrective steps suffice
// because inputs are in [0, p) so sum is in [0, 2p) ⊂ [0, 2^65).
inline ulong gl_add(ulong a, ulong b) {
    ulong s = a + b;
    // carry from a+b
    ulong carry = (s < a) ? 1UL : 0UL;
    // adjust: if carry, add CQ; if that also wraps, add CQ again.
    s += carry * GL_CQ;
    ulong carry2 = (carry != 0 && s < GL_CQ) ? 1UL : 0UL;
    s += carry2 * GL_CQ;
    return s;
}

// gl_sub: fully reduced result in [0, p)
// Algorithm: s = a - b; on borrow (s > a), subtract CQ (≡ add p mod 2^64).
// Two subtractive steps suffice for same reason.
inline ulong gl_sub(ulong a, ulong b) {
    ulong s = a - b;
    // borrow from a-b
    ulong borrow = (s > a) ? 1UL : 0UL;
    ulong prev = s;
    s -= borrow * GL_CQ;
    ulong borrow2 = (borrow != 0 && s > prev) ? 1UL : 0UL;
    s -= borrow2 * GL_CQ;
    return s;
}

// gl_mul: LAZY reduce — result in [0, 2p), NOT canonicalized.
//
// Port of mul_scalar (goldilocks_simd_neon.hpp:143-161) to MSL.
// Uses metal::mulhi(a,b) for high 64 bits of 64x64 product.
//
// Algorithm (identical to NEON path — deliberately omits final if r>=p):
//   prod = a * b  (128-bit conceptual)
//   c_lo = low64(prod), c_hi = high64(prod)
//   c_hi_lo = c_hi[31:0], c_hi_hi = c_hi[63:32]
//   s  = c_lo - c_hi              -- may borrow
//   s2 = s + c_hi_lo * 2^32       -- may carry
//   s3 = s2 + c_hi_hi * CQ        -- may carry  (CQ = 2^32-1)
//   adj in {-1,0,1,2}: r = s3 + adj*CQ
//   NO final if (r >= p) r -= p   -- lazy reduce
inline ulong gl_mul(ulong a, ulong b) {
    ulong c_lo  = a * b;
    ulong c_hi  = metal::mulhi(a, b);
    ulong c_hi_lo = c_hi & GL_CQ;          // lower 32 bits of c_hi
    ulong c_hi_hi = c_hi >> 32;             // upper 32 bits of c_hi

    ulong s  = c_lo - c_hi;
    bool borrow = s > c_lo;

    ulong s2 = s + (c_hi_lo << 32);
    bool carry2 = s2 < s;

    ulong s3 = s2 + c_hi_hi * GL_CQ;
    bool carry3 = s3 < s2;

    int adj = (int)carry2 + (int)carry3 - (int)borrow;
    ulong r;
    if (adj >= 0) {
        r = s3 + (ulong)adj * GL_CQ;
    } else {
        r = s3 - GL_CQ;
    }
    // INTENTIONALLY no: if (r >= GL_PRIME) r -= GL_PRIME;
    return r;
}

// gl_canonicalize: [0, 2p) -> [0, p)
// Call only at kernel boundaries / MTLBuffer write.
inline ulong gl_canonicalize(ulong a) {
    return (a >= GL_PRIME) ? (a - GL_PRIME) : a;
}
