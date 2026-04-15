// ntt.metal — NTT kernels for Goldilocks field
//
// Implements:
//   kernel ntt_reverse_permutation  — bit-reverse permutation (in-place)
//   kernel ntt_butterfly_phase      — one Cooley-Tukey phase (in-place)
//   kernel intt_scale               — multiply each element by 1/N (inverse NTT scaling)
//
// Buffer layout: row-major, stride = ncols.
//   element[row][col] = buf[row * ncols + col]
//
// CPU reference: BR() at ntt_goldilocks.cpp:6-13
//   Butterfly math: t = w * a[off1]; a[off2] = u + t; a[off1] = u - t
//   where u = a[off2].

#include <metal_stdlib>
#include "field.metal"
using namespace metal;

// reverse_bits32: full 32-bit bit-reversal of x
// Then shift right by (32 - domainPow) to get the log2(N)-bit reversal.
// Matches BR() in ntt_goldilocks.cpp:6-13 exactly.
inline uint reverse_bits32(uint x) {
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00u) >> 8)  | ((x & 0x00FF00FFu) << 8);
    x = ((x & 0xF0F0F0F0u) >> 4)  | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x & 0xCCCCCCCCu) >> 2)  | ((x & 0x33333333u) << 2);
    x = ((x & 0xAAAAAAAAu) >> 1)  | ((x & 0x55555555u) << 1);
    return x;
}

// ---- kernel: ntt_reverse_permutation ---------------------------------------
// One thread per (element-pair, col) — but for simplicity, dispatch one thread
// per flat index [0, domain_size * ncols) and let threads with src < dst swap.
//
// tid = thread index in [0, domain_size)  (one per row, all cols handled)
// domainPow = log2(domain_size)
kernel void ntt_reverse_permutation(
    device ulong*   buf       [[ buffer(0) ]],
    constant uint&  domainPow [[ buffer(1) ]],
    constant uint&  ncols     [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint domain_size = 1u << domainPow;
    if (tid >= domain_size) return;

    uint rev = reverse_bits32(tid) >> (32 - domainPow);
    if (rev <= tid) return;  // only process each pair once

    uint src = tid  * ncols;
    uint dst = rev  * ncols;
    for (uint c = 0; c < ncols; c++) {
        ulong tmp     = buf[src + c];
        buf[src + c]  = buf[dst + c];
        buf[dst + c]  = tmp;
    }
}

// ---- kernel: ntt_butterfly_phase -------------------------------------------
// One thread per (butterfly pair index, col).
// Implements one level s of the DIT Cooley-Tukey NTT.
//
// Parameters:
//   buf         = data buffer [domain_size * ncols]
//   twiddles    = precomputed twiddle factors, twiddles[j] for j in [0, mdiv2)
//   ncols       = number of columns (stride)
//   domain_size = 2^domainPow
//   s           = current NTT level in [1, domainPow]
//
// Thread grid: dispatch (domain_size/2 * ncols) threads.
// tid maps to: pair_idx = tid / ncols, col = tid % ncols
//
// Butterfly:
//   m     = 2^s;  mdiv2 = m/2
//   Group g = pair_idx / mdiv2
//   Intra-group offset j = pair_idx % mdiv2
//   off2 = g*m + j          (upper element in butterfly)
//   off1 = g*m + j + mdiv2  (lower element)
//   w  = twiddles[j]
//   u  = buf[off2 * ncols + col]
//   t  = gl_mul(w, buf[off1 * ncols + col])
//   buf[off2 * ncols + col] = gl_add(u, t)   -- no lazy needed, gl_add canonicalizes
//   buf[off1 * ncols + col] = gl_sub(u, t)
// The kernel reads from a single, instance-wide twiddles buffer that holds
// the full roots[0 .. 2^s_global) array. For phase s, the twiddle at logical
// index j is roots[j << (s_global - s)] (matches CPU root(s, j) at
// ntt_goldilocks.hpp:169-172). `roots_stride_shift` is the per-call
// stride = s_global - s. This removes the per-phase twiddle buffer
// allocation + upload that dominated small-N and multi-column runs.
kernel void ntt_butterfly_phase(
    device ulong*         buf                  [[ buffer(0) ]],
    device const ulong*   twiddles             [[ buffer(1) ]],
    constant uint&        ncols                [[ buffer(2) ]],
    constant uint&        domain_size          [[ buffer(3) ]],
    constant uint&        s                    [[ buffer(4) ]],
    constant uint&        roots_stride_shift   [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint half_n = domain_size >> 1;
    uint total = half_n * ncols;
    if (tid >= total) return;

    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;

    uint m     = 1u << s;
    uint mdiv2 = m >> 1;
    uint g     = pair_idx / mdiv2;
    uint j     = pair_idx % mdiv2;

    uint off2 = (g * m + j) * ncols + col;
    uint off1 = (g * m + j + mdiv2) * ncols + col;

    ulong w = twiddles[j << roots_stride_shift];
    ulong u = buf[off2];
    ulong t = gl_mul(w, buf[off1]);

    buf[off2] = gl_add(u, t);
    buf[off1] = gl_sub(u, t);
}

// ---- kernel: ntt_rev_butterfly_s1 -----------------------------------------
// Fused kernel: reads input in natural order, writes bit-reverse-permuted
// AND phase-1 butterfly output. Replaces two separate passes
// (ntt_reverse_permutation + ntt_butterfly_phase with s=1) with a single
// pass, saving one full N-element read+write cycle plus one commit.
//
// Why this works: at phase s=1, mdiv2=1, so each butterfly pair uses
// twiddle index j=0, and root(1, 0) = roots[0] = 1 — the twiddle
// multiplication is a no-op. The s=1 butterfly reduces to just
// gl_add/gl_sub of two bit-reversed-position inputs.
//
// Requires src ≠ dst (out-of-place). Caller guarantees via
// the usual memcpy(dst, src) path, but reads from src directly here
// to save one memcpy pass.
//
// Dispatch: (domain_size / 2) × ncols threads. Thread tid handles one
// (butterfly-pair, column) pair.
kernel void ntt_rev_butterfly_s1(
    device const ulong* src        [[ buffer(0) ]],  // input (natural order)
    device       ulong* dst        [[ buffer(1) ]],  // output (perm + s=1 butterfly)
    constant     uint&  domain_pow [[ buffer(2) ]],
    constant     uint&  ncols      [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint half_n = (1u << domain_pow) >> 1;
    uint total  = half_n * ncols;
    if (tid >= total) return;

    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;

    // After bit-reversal, the butterfly pair at natural positions
    // (2*pair_idx, 2*pair_idx+1) comes from src positions
    //   reverse_bits(2*pair_idx)   >> (32 - domain_pow)
    //   reverse_bits(2*pair_idx+1) >> (32 - domain_pow)
    uint shift = 32u - domain_pow;
    uint src_a = reverse_bits32(pair_idx * 2u)     >> shift;
    uint src_b = reverse_bits32(pair_idx * 2u + 1u) >> shift;

    ulong u = src[src_a * ncols + col];
    ulong t = src[src_b * ncols + col];  // twiddle = 1 at s=1, no mul needed

    dst[(pair_idx * 2u)     * ncols + col] = gl_add(u, t);
    dst[(pair_idx * 2u + 1u) * ncols + col] = gl_sub(u, t);
}

// ---- kernel: intt_reorder --------------------------------------------------
// Inverse-NTT index permutation: out[(N - i) % N] = in[i].
// CPU reference: NTT_iters uses intt_idx(i,N) = (N - i) % N before INTT scale
// (ntt_goldilocks.cpp:175,188; ntt_goldilocks.hpp:38-46).
//
// Implemented in-place as a swap between index i and (N - i) for i in [1, N/2),
// once per row. Element 0 is fixed (N % N = 0). If N is even, element N/2 is
// also fixed (N - N/2 = N/2). Dispatch N/2 threads; thread tid handles pair
// (tid+1, N - (tid+1)) for tid in [0, N/2 - 1).
kernel void intt_reorder(
    device ulong*    buf           [[ buffer(0) ]],
    constant uint&   domain_size   [[ buffer(1) ]],
    constant uint&   ncols         [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint i = tid + 1;
    uint lo = i;
    uint hi = domain_size - i;
    if (lo >= hi) return;  // only swap once per pair; stop at the middle
    for (uint c = 0; c < ncols; c++) {
        ulong a = buf[lo * ncols + c];
        ulong b = buf[hi * ncols + c];
        buf[lo * ncols + c] = b;
        buf[hi * ncols + c] = a;
    }
}

// ---- kernel: intt_scale ----------------------------------------------------
// One thread per flat element index [0, domain_size * ncols).
// Multiplies each element by the precomputed scalar inv_n = 1/domain_size mod p.
// Used as the final step of the inverse NTT (after intt_reorder).
kernel void intt_scale(
    device ulong*       buf     [[ buffer(0) ]],
    constant ulong&     inv_n   [[ buffer(1) ]],
    constant uint&      count   [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= count) return;
    buf[tid] = gl_canonicalize(gl_mul(buf[tid], inv_n));
}
