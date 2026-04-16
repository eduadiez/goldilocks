// poseidon_metal.mm — Metal implementation of merkletree_metal.
//
// Mirrors the structure of PoseidonGoldilocks::merkletree_seq:
//   1. Dispatch merkle_leaves to compute leaf hashes (one thread per row).
//   2. Level-by-level dispatch of merkle_parents until root is reached.
//
// Buffer strategy:
//   - For `input`: try zero-copy alias (newBufferWithBytesNoCopy) if ptr is
//     16 KB page-aligned. Otherwise copy in. No readback needed for input.
//   - For `tree`:  same alias attempt; if a copy was made, readback via
//     memcpy from buffer.contents into caller's tree pointer after all work.
//
// OMP constraint: do NOT call this from inside an OMP parallel region.
//   The GPU parallelises all leaf hashing internally; the host loop over
//   tree levels is sequential on the calling thread.
//
// Build: clang++ -std=c++17 -fobjc-arc -ObjC++ -DGOLDILOCKS_HAS_METAL \
//               -framework Metal -framework Foundation -I../../src

#ifdef GOLDILOCKS_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <vector>

#include <thread>
#include <omp.h>

#include "../goldilocks_base_field.hpp"
#include "../poseidon_goldilocks.hpp"
#include "metal_context.hpp"
#include "goldilocks_metal.hpp"  // declares goldilocks_metal::merkletree_metal

// A/B switch for leaf kernel. 0 = one-thread-per-row (default; shipped).
// 1 = SIMD-cooperative (experimental, being A/B'd). Set via env var
// GOLDILOCKS_METAL_COOP by `bench_merkle`, or left at 0 for normal callers.
// Weak symbol so test harnesses can override without linker games.
// A/B selector for leaf kernel:
//   0 = row-per-thread (default, shipped)
//   1 = SIMD-cooperative
//   2 = 2-rows-per-thread (ILP variant)
//   3 = transpose + column-major (coalesced memory reads)
extern "C" { __attribute__((weak)) int g_merkle_use_simd_coop = 0; }

// Constants mirroring poseidon_goldilocks.hpp / merklehash_goldilocks.hpp
static constexpr uint32_t HASH_SIZE_ = 4;   // CAPACITY = HASH_SIZE = 4 elements
static constexpr uint32_t RATE_      = 8;   // RATE = 8 elements

namespace goldilocks_metal {

void merkletree_metal(Goldilocks::Element* tree,
                      Goldilocks::Element* input,
                      uint64_t num_cols,
                      uint64_t num_rows)
{
    @autoreleasepool {
        if (num_rows == 0) return;

        MetalCtxHandle ctx = metal_context_get();

        // -------------------------------------------------------------------
        // numElementsTree = (2*num_rows - 1) * HASH_SIZE
        // leaf hashes occupy tree[0 .. num_rows*4)
        // parent levels are stacked above
        // -------------------------------------------------------------------
        uint64_t numElementsTree = (2 * num_rows - 1) * HASH_SIZE_;
        size_t tree_bytes  = numElementsTree * sizeof(Goldilocks::Element);
        size_t input_bytes = num_rows * num_cols * sizeof(Goldilocks::Element);

        // -------------------------------------------------------------------
        // Allocate / alias buffers
        // -------------------------------------------------------------------
        int input_is_copy = 0;
        int tree_is_copy  = 0;
        MetalBufHandle input_buf = NULL;
        MetalBufHandle tree_buf  = NULL;

        // Input buffer (read-only by GPU): try zero-copy alias if num_cols > 0.
        if (num_cols > 0 && input != NULL) {
            input_buf = metal_buf_alias(ctx, (void*)input, input_bytes, &input_is_copy);
        } else {
            // num_cols == 0: merkle_leaves kernel still runs but reads nothing.
            // Allocate a tiny dummy buffer (1 byte minimum).
            input_buf = metal_buf_alloc(ctx, sizeof(uint64_t));
        }

        // Tree buffer (read-write by GPU and CPU): alias or alloc.
        tree_buf = metal_buf_alias(ctx, (void*)tree, tree_bytes, &tree_is_copy);

        // -------------------------------------------------------------------
        // Phase 1: merkle_leaves — one thread per leaf row.
        // kernel signature: (inp, tree, ncols, dim)
        // dim is always 1 for merkletree (no cubic extension here).
        // -------------------------------------------------------------------
        uint32_t ncols32 = (uint32_t)num_cols;
        uint32_t dim32   = 1;
        uint32_t nrows32 = (uint32_t)num_rows;

        switch (g_merkle_use_simd_coop) {
            case 1:
                metal_dispatch_merkle_leaves_simd(ctx, input_buf, tree_buf,
                                                  ncols32, dim32, nrows32);
                break;
            case 2:
                metal_dispatch_merkle_leaves_x2(ctx, input_buf, tree_buf,
                                                 ncols32, dim32, nrows32);
                break;
            case 3: {
                // Transpose input to column-major, then run coalesced-read
                // kernel. Extra pass but unlocks simdgroup memory coalescing
                // for the pod12 absorb loop (8 reads per thread × 16
                // iterations per row, adjacent threads → adjacent addresses).
                MetalBufHandle inp_cm = metal_buf_alloc(ctx, input_bytes);
                metal_dispatch_transpose_rowmajor(ctx, input_buf, inp_cm,
                                                   nrows32, ncols32);
                metal_dispatch_merkle_leaves_cm(ctx, inp_cm, tree_buf,
                                                 ncols32, dim32, nrows32);
                metal_buf_release(inp_cm);
                break;
            }
            case 4:
                // Fused-tile: cooperative threadgroup load + pod12 in one
                // kernel. Avoids the separate transpose pass's alloc and
                // round-trip through global memory.
                metal_dispatch_merkle_leaves_tg(ctx, input_buf, tree_buf,
                                                 ncols32, dim32, nrows32);
                break;
            default:
                metal_dispatch_merkle_leaves(ctx, input_buf, tree_buf,
                                              ncols32, dim32, nrows32);
                break;
        }

        // -------------------------------------------------------------------
        // Phase 2: merkle_parents — one GPU dispatch per tree level.
        // Mirrors the CPU loop in merkletree_seq:
        //   pending     = num_rows (children at current level)
        //   nextN       = ceil(pending / 2) = number of parent nodes to write
        //   nextIndex   = flat element offset to START of children at this level
        // -------------------------------------------------------------------
        // All tree levels in one command buffer (one waitUntilCompleted).
        metal_dispatch_merkle_parents_all_levels(ctx,
                                                  tree_buf,
                                                  (uint32_t)num_rows);

        // -------------------------------------------------------------------
        // Readback if tree buffer was a copy (not a zero-copy alias).
        // -------------------------------------------------------------------
        if (tree_is_copy) {
            void* gpu_ptr = metal_buf_contents(tree_buf);
            std::memcpy(tree, gpu_ptr, tree_bytes);
        }

        // Release buffer handles (decrements ARC-retained count).
        metal_buf_release(input_buf);
        metal_buf_release(tree_buf);
    }  // @autoreleasepool
}

// ---------------------------------------------------------------------------
// merkletree_metal_batch — build `count` Merkle trees in one GPU submission.
//
// Why this helps: the GPU queue runs command buffers serially, but WITHIN
// one command buffer Metal's hazard tracker allows dispatches from different
// compute encoders to overlap if they touch different resources. Putting
// each tree in its OWN encoder inside one big command buffer lets the GPU
// start tree N+1's leaf kernel while tree N's parent loop is still running.
//
// All trees must share (num_cols, num_rows). Same bit-exact contract as
// merkletree_metal.
void merkletree_metal_batch(Goldilocks::Element** trees,
                            Goldilocks::Element** inputs,
                            uint64_t count,
                            uint64_t num_cols,
                            uint64_t num_rows)
{
    if (count == 0 || num_rows == 0) return;
    @autoreleasepool {
        MetalCtxHandle ctx = metal_context_get();

        uint64_t numElementsTree = (2 * num_rows - 1) * HASH_SIZE_;
        size_t   tree_bytes      = numElementsTree * sizeof(Goldilocks::Element);
        size_t   input_bytes     = num_rows * num_cols * sizeof(Goldilocks::Element);

        std::vector<MetalBufHandle> in_bufs(count, nullptr);
        std::vector<MetalBufHandle> tree_bufs(count, nullptr);
        std::vector<int>            in_is_copy(count, 0);
        std::vector<int>            tree_is_copy(count, 0);

        for (uint64_t i = 0; i < count; i++) {
            if (num_cols > 0 && inputs[i] != nullptr) {
                in_bufs[i] = metal_buf_alias(ctx, (void*)inputs[i],
                                              input_bytes, &in_is_copy[i]);
            } else {
                in_bufs[i] = metal_buf_alloc(ctx, sizeof(uint64_t));
            }
            tree_bufs[i] = metal_buf_alias(ctx, (void*)trees[i],
                                            tree_bytes, &tree_is_copy[i]);
        }

        uint32_t ncols32 = (uint32_t)num_cols;
        uint32_t dim32   = 1;
        uint32_t nrows32 = (uint32_t)num_rows;

        metal_dispatch_merkletree_batch(ctx,
                                         in_bufs.data(),
                                         tree_bufs.data(),
                                         (uint32_t)count,
                                         ncols32, dim32, nrows32);

        // Readback copy-fallback trees and release handles.
        for (uint64_t i = 0; i < count; i++) {
            if (tree_is_copy[i]) {
                void* gpu_ptr = metal_buf_contents(tree_bufs[i]);
                std::memcpy(trees[i], gpu_ptr, tree_bytes);
            }
            metal_buf_release(in_bufs[i]);
            metal_buf_release(tree_bufs[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Page-aligned allocator for zero-copy MTLBuffer aliasing.
//
// MTLDevice `newBufferWithBytesNoCopy` requires the backing pointer to be
// aligned to a VM page boundary (16 KB on Apple Silicon). When callers
// allocate via `new Element[]` or `malloc`, alignment is typically 16 B
// — far too small — so the Metal bridge falls back to a `newBufferWithBytes`
// copy + readback. For large inputs/outputs the memcpy dominates.
//
// allocate_aligned_elements returns page-aligned memory so the bridge takes
// the zero-copy path automatically.
//
// Implementation: posix_memalign with 16 KB alignment, rounded up to the
// nearest page (posix_memalign doesn't require size to be a multiple of
// alignment on modern libc, but some older libs do).
// ---------------------------------------------------------------------------
// merkletree_metal_leaves_only — GPU leaf hashing for a contiguous row slice.
//
// `tree_out` must point to the slice start inside the caller's full tree
// buffer. `input` must point to the matching row-major input slice. We alias
// (or copy-fallback) both and run ONLY the leaves kernel over nrows_partial
// rows. Parent levels are the caller's responsibility (see
// merkletree_metal_parents_only).
//
// This is the GPU half of the hybrid CPU+GPU Merkle dispatcher.
// ---------------------------------------------------------------------------
void merkletree_metal_leaves_only(Goldilocks::Element* tree_out,
                                  Goldilocks::Element* input,
                                  uint64_t ncols,
                                  uint64_t nrows_partial)
{
    @autoreleasepool {
        if (nrows_partial == 0) return;
        MetalCtxHandle ctx = metal_context_get();

        size_t out_bytes   = nrows_partial * HASH_SIZE_ * sizeof(Goldilocks::Element);
        size_t input_bytes = nrows_partial * ncols      * sizeof(Goldilocks::Element);

        int in_is_copy = 0, out_is_copy = 0;
        MetalBufHandle input_buf = NULL;
        MetalBufHandle out_buf   = NULL;

        if (ncols > 0 && input != NULL) {
            input_buf = metal_buf_alias(ctx, (void*)input, input_bytes, &in_is_copy);
        } else {
            input_buf = metal_buf_alloc(ctx, sizeof(uint64_t));
        }
        out_buf = metal_buf_alias(ctx, (void*)tree_out, out_bytes, &out_is_copy);

        uint32_t ncols32 = (uint32_t)ncols;
        uint32_t dim32   = 1;
        uint32_t nrows32 = (uint32_t)nrows_partial;

        // Always the default row kernel — the cooperative / x2 / cm / tg
        // variants don't offer a performance win at production sizes and
        // would complicate the hybrid handshake needlessly.
        metal_dispatch_merkle_leaves(ctx, input_buf, out_buf,
                                      ncols32, dim32, nrows32);

        if (out_is_copy) {
            void* gpu_ptr = metal_buf_contents(out_buf);
            std::memcpy(tree_out, gpu_ptr, out_bytes);
        }
        metal_buf_release(input_buf);
        metal_buf_release(out_buf);
    }
}

// ---------------------------------------------------------------------------
// merkletree_metal_parents_only — GPU parent-level reduction over leaves
// already present at tree[0 .. nrows_total × HASH_SIZE). After this call the
// tree buffer contains the full Merkle tree up to the root.
// ---------------------------------------------------------------------------
void merkletree_metal_parents_only(Goldilocks::Element* tree,
                                   uint64_t nrows_total)
{
    @autoreleasepool {
        if (nrows_total == 0) return;
        MetalCtxHandle ctx = metal_context_get();

        uint64_t numElementsTree = (2 * nrows_total - 1) * HASH_SIZE_;
        size_t tree_bytes = numElementsTree * sizeof(Goldilocks::Element);

        int is_copy = 0;
        MetalBufHandle tree_buf = metal_buf_alias(
            ctx, (void*)tree, tree_bytes, &is_copy);

        metal_dispatch_merkle_parents_all_levels(
            ctx, tree_buf, (uint32_t)nrows_total);

        if (is_copy) {
            void* gpu_ptr = metal_buf_contents(tree_buf);
            std::memcpy(tree, gpu_ptr, tree_bytes);
        }
        metal_buf_release(tree_buf);
    }
}

// ---------------------------------------------------------------------------
// merkletree_hybrid — CPU NEON + GPU Metal concurrent Merkle build.
//
// Strategy: leaf-row split.
//   K = floor(nrows × cpu_fraction)
//   [0, K)          → CPU NEON (linear_hash_neon, OMP-parallel)
//   [K, nrows)      → GPU Metal (merkle_leaves kernel)
//   ↓ both concurrently via std::thread ↓
//   Tree reduction  → GPU (merkle_parents_all_levels)
//
// Unified memory makes this zero-copy: both engines write disjoint regions
// of the same `tree` buffer. Contention-probe measurements (see
// bench_contention_merkle.mm) show 0.99 overlap efficiency at 262k rows,
// so the concurrent step is ~max(T_cpu_share, T_gpu_share) rather than
// T_cpu_share + T_gpu_share.
//
// OMP constraint: as with every entry in goldilocks_metal::, do NOT call
// this from inside a pre-existing OMP parallel region. The function itself
// spawns an OMP block internally; nesting would oversubscribe.
// ---------------------------------------------------------------------------
void merkletree_hybrid(Goldilocks::Element* tree,
                       Goldilocks::Element* input,
                       uint64_t ncols,
                       uint64_t nrows,
                       double cpu_fraction)
{
    if (nrows == 0) return;
    if (cpu_fraction < 0.0) cpu_fraction = 0.0;
    if (cpu_fraction > 1.0) cpu_fraction = 1.0;

    uint64_t K = (uint64_t)((double)nrows * cpu_fraction);

    // Degenerate splits → fall back to single-engine code paths.
    if (K == 0) {
        merkletree_metal(tree, input, ncols, nrows);
        return;
    }
    if (K >= nrows) {
        PoseidonGoldilocks::merkletree_neon(tree, input, ncols, nrows);
        return;
    }

    // GPU helper thread: leaves for rows [K, nrows).
    std::thread gpu_thread([&]{
        merkletree_metal_leaves_only(
            tree  + K * HASH_SIZE_,
            input + K * ncols,
            ncols,
            nrows - K);
    });

    // CPU leaves for rows [0, K) on the calling thread, OMP-parallel.
    //
    // Two choices matter for throughput on M4 Pro:
    //   1. Use linear_hash_neon_pair (2-row ILP) instead of per-row — the
    //      same inner loop that merkletree_neon uses. 1.3× speedup.
    //   2. Cap the thread count at the P-core count (10 on M4 Pro).
    //      libomp will happily schedule on E-cores, but they run ~3× slower
    //      per core and with schedule(static) they become stragglers that
    //      dilate the whole wall time. Empirically 10 P-core threads beats
    //      14 P+E-core threads by ~2× on this workload.
    //   Also reserve one core for the gpu_thread that's driving the Metal
    //   command buffer.
    int cpu_threads = omp_get_max_threads();
    if (cpu_threads > 10) cpu_threads = 10;   // P-core cap
    if (cpu_threads > 1)  cpu_threads -= 1;   // reserve for gpu_thread

    #pragma omp parallel for schedule(static) num_threads(cpu_threads)
    for (uint64_t r = 0; r < K; r += 2) {
        if (r + 1 < K) {
            PoseidonGoldilocks::linear_hash_neon_pair(
                &tree[ r      * HASH_SIZE_], &input[ r      * ncols],
                &tree[(r + 1) * HASH_SIZE_], &input[(r + 1) * ncols],
                ncols);
        } else {
            PoseidonGoldilocks::linear_hash_neon(
                &tree[r * HASH_SIZE_],
                &input[r * ncols],
                ncols);
        }
    }

    gpu_thread.join();

    // All leaves now materialized across tree[0 .. nrows*HASH_SIZE_).
    merkletree_metal_parents_only(tree, nrows);
}

Goldilocks::Element* allocate_aligned_elements(uint64_t n) {
    if (n == 0) return nullptr;
    constexpr size_t PAGE = 16384;
    size_t bytes = n * sizeof(Goldilocks::Element);
    size_t aligned = (bytes + PAGE - 1) & ~(PAGE - 1);
    void* p = nullptr;
    if (posix_memalign(&p, PAGE, aligned) != 0) return nullptr;
    return reinterpret_cast<Goldilocks::Element*>(p);
}

void free_aligned(void* ptr) {
    free(ptr);
}

}  // namespace goldilocks_metal

#endif // GOLDILOCKS_HAS_METAL
