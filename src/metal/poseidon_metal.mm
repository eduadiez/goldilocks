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

#include "../goldilocks_base_field.hpp"
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

}  // namespace goldilocks_metal

#endif // GOLDILOCKS_HAS_METAL
