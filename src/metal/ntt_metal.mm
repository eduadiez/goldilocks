// ntt_metal.mm — Metal implementation of NTT_Metal.
//
// Implements the Cooley-Tukey DIT NTT on the GPU using three kernel types:
//   1. ntt_reverse_permutation  — bit-reverse permute the source into dst
//   2. ntt_butterfly_phase      — one level s of butterfly per dispatch
//   3. intt_scale               — multiply all elements by 1/N (inverse only)
//
// Algorithm mirrors NTT_Goldilocks::NTT_iters (ntt_goldilocks.cpp), plain path
// (no coset/extend — deferred per SDD).
//
// Twiddle factors:
//   The GPU butterfly kernel for level s needs mdiv2 = 2^(s-1) twiddle factors.
//   These are roots[j << (ctx_s - s)] for j in [0, mdiv2), which are exactly
//   the first mdiv2 entries of ntt_ctx->root(s, j).
//   We stage the FULL roots array (length 1 << ctx_s) into a single MTLBuffer
//   keyed by roots_len, then the kernel indexes into it the same way root() does
//   by passing the full roots buffer and having the kernel use j directly
//   (kernel uses twiddles[j] where twiddles is roots shifted by (ctx_s-s) stride).
//
//   Simpler approach (used here): for each level s, extract the relevant slice
//   into the full roots buffer and pass the full buffer — the kernel reads
//   twiddles[j] and we set the twiddles buffer pointer to roots + offset where
//   offset = 0 and the kernel's stride into the buffer is j << (s - domainPow)...
//
//   Actually: ntt_butterfly_phase kernel uses twiddles[j] directly for j in
//   [0, mdiv2). root(domainPow, j) = roots[j << (ctx_s - domainPow)].
//   So for level s: twiddle[j] = roots[j << (ctx_s - s)].
//   The simplest correct approach: for each phase s, pass an offset into the
//   full roots buffer equal to byte_offset = 0, and let the kernel index:
//     twiddles[j * (1 << (ctx_s - s))]
//   BUT the kernel as written uses twiddles[j] without a stride parameter.
//
//   Resolution: pre-build a per-level twiddle slice using metal_twiddle_buffer
//   keyed by (domainPow << 32 | s), or — simpler — compute the slice on the CPU
//   and upload as a temporary buffer. We choose: for each call, build a full
//   flattened twiddles array of length (domain_size/2) that is EXACTLY the roots
//   indexed as root(domainPow, j) = roots[j << (s_global - domainPow)] for
//   j in [0, domain_size/2). This is a one-time upload at NTT_Metal call time.
//
//   Then for each phase s, the kernel needs mdiv2 = 2^(s-1) twiddle values:
//     w[j] = roots[j << (s_global - s)]  for j in [0, 2^(s-1))
//   We can represent this as a stride-based read into the full roots buffer:
//     offset_elems = 0, stride = 1 << (s_global - s)
//   But the kernel doesn't support stride. So we pack per-level slice buffers
//   at buffer creation time (one per level) into a single staging buffer, and
//   pass the slice as a buffer with byte_offset.
//
//   FINAL CHOICE (simplest, provably correct):
//   Allocate one buffer of length (domain_size/2) from the full roots array.
//   For level s, the twiddle for index j is:
//       twiddle_j = roots[j << (s_global - s)]
//   We allocate a single "all-twiddles" buffer of domain_size/2 elements where
//   element[j] = roots[j << (s_global - domainPow)] — i.e. root(domainPow, j).
//   For level s < domainPow, the needed twiddles are a SUBSET: indices 0,2,...
//   strided by 2^(domainPow-s). The kernel is dispatched with mdiv2 threads and
//   uses twiddles[j]. We need twiddles[j] = roots[j << (s_global - s)].
//
//   We do per-phase: build a CPU-side array of mdiv2 uint64_t values from
//   ntt_ctx->roots_ptr() and upload as a fresh buffer per phase.
//   This is O(N/2) copies total across all phases = O(N log N) ops — acceptable.
//
// OMP constraint: NTT_Metal must not be called from within an OMP parallel region.
//
// Build: clang++ -std=c++17 -fobjc-arc -ObjC++ -DGOLDILOCKS_HAS_METAL \
//               -framework Metal -framework Foundation -I../../src

#ifdef GOLDILOCKS_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "../goldilocks_base_field.hpp"
#include "../ntt_goldilocks.hpp"
#include "metal_context.hpp"
#include "goldilocks_metal.hpp"  // declares goldilocks_metal::NTT_Metal

namespace goldilocks_metal {

void NTT_Metal(Goldilocks::Element* dst,
               Goldilocks::Element* src,
               uint64_t size,
               uint64_t ncols,
               NTT_Goldilocks* ntt_ctx,
               bool inverse)
{
    @autoreleasepool {
        if (size == 0 || ncols == 0) return;

        MetalCtxHandle ctx = metal_context_get();

        uint32_t domainPow = 0;
        {
            uint64_t s = size;
            while (s > 1) { s >>= 1; domainPow++; }
        }

        // Globals from ntt_ctx
        uint32_t s_global = ntt_ctx->get_s();
        const Goldilocks::Element* roots_raw = ntt_ctx->roots_ptr();
        uint64_t rlen = ntt_ctx->roots_len();  // = 1 << s_global
        (void)rlen;

        size_t data_bytes = size * ncols * sizeof(Goldilocks::Element);
        uint32_t ncols32  = (uint32_t)ncols;
        uint32_t domain32 = (uint32_t)size;

        int dst_is_copy = 0;
        MetalBufHandle dst_buf = metal_buf_alias(ctx, (void*)dst, data_bytes, &dst_is_copy);

        // -------------------------------------------------------------------
        // Fused-path (preferred): src ≠ dst. Skip the memcpy + reverse +
        // phase-1 butterfly (three passes over the data) and do all three
        // in one pass via ntt_rev_butterfly_s1.
        //
        // Fallback: src == dst (in-place). memcpy is a no-op, we must
        // run the legacy reverse-permutation + full phase loop.
        // -------------------------------------------------------------------
        MetalBufHandle tw_buf = metal_twiddle_buffer(
            ctx,
            reinterpret_cast<const uint64_t*>(roots_raw),
            rlen);

        if (dst != src && domainPow >= 1) {
            // Alias src read-only; the fused kernel reads natural-order src
            // and writes bit-reverse-permuted + s=1-butterflied dst in one pass.
            int src_is_copy = 0;
            MetalBufHandle src_buf = metal_buf_alias(ctx, (void*)src,
                                                      data_bytes, &src_is_copy);

            metal_dispatch_ntt_rev_butterfly_s1(
                ctx, src_buf, dst_buf, domainPow, ncols32);

            // Continue from phase s=2.
            metal_dispatch_ntt_butterfly_all_phases(
                ctx, dst_buf, tw_buf,
                ncols32, domain32,
                /*start_s=*/2, domainPow, s_global);

            metal_buf_release(src_buf);
        } else {
            // In-place path (src == dst) or trivial N=1.
            if (dst != src) std::memcpy(dst, src, data_bytes);
            metal_dispatch_ntt_reverse_permutation(ctx, dst_buf, domainPow, ncols32);
            metal_dispatch_ntt_butterfly_all_phases(
                ctx, dst_buf, tw_buf,
                ncols32, domain32,
                /*start_s=*/1, domainPow, s_global);
        }

        // -------------------------------------------------------------------
        // 5. Inverse NTT: scale each element by 1/N mod p.
        //    powTwoInv[domainPow] holds the precomputed value; access via
        //    root accessor can't get it — use the public root() interface:
        //    Actually powTwoInv is not exposed. Compute 1/N using Goldilocks::inv.
        // -------------------------------------------------------------------
        if (inverse) {
            // INTT = forward-NTT butterflies + index permutation (N-i) % N + scale 1/N.
            // Fused reorder+scale kernel does both in one pass/commit vs the
            // previous two-dispatch approach (see ntt_reorder_scale in
            // ntt.metal).
            Goldilocks::Element inv_n = Goldilocks::inv(Goldilocks::fromU64(size));
            uint64_t inv_n_u64 = Goldilocks::toU64(inv_n);
            metal_dispatch_intt_reorder_scale(ctx, dst_buf,
                                                domain32, ncols32, inv_n_u64);
        }

        // -------------------------------------------------------------------
        // 6. Readback if buffer was a copy.
        // -------------------------------------------------------------------
        if (dst_is_copy) {
            void* gpu_ptr = metal_buf_contents(dst_buf);
            std::memcpy(dst, gpu_ptr, data_bytes);
        }

        metal_buf_release(dst_buf);
    }  // @autoreleasepool
}

}  // namespace goldilocks_metal

#endif // GOLDILOCKS_HAS_METAL
