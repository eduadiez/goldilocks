#pragma once
#ifdef GOLDILOCKS_HAS_METAL
#include <cstdint>
#include "../goldilocks_base_field.hpp"
class NTT_Goldilocks;  // forward declaration — avoids pulling in ntt_goldilocks.hpp

// THREADING CONTRACT: the bridge entries below MUST be invoked from the
// main/submitting thread — NEVER from inside a `#pragma omp parallel` region.
// Metal framework objects require an @autoreleasepool on the calling thread;
// OMP worker threads do not have one and would leak Obj-C objects per iter.
// The bridge already wraps each entry in @autoreleasepool, but only the
// single thread that called the entry is protected.
namespace goldilocks_metal {
    void merkletree_metal(Goldilocks::Element* tree, Goldilocks::Element* input,
                          uint64_t num_cols, uint64_t num_rows);
    void NTT_Metal(Goldilocks::Element* dst, Goldilocks::Element* src,
                   uint64_t size, uint64_t ncols, NTT_Goldilocks* ntt_ctx, bool inverse);

    // Allocate an array of `n` Goldilocks::Element on a 16KB-page-aligned
    // boundary. Buffers passed to merkletree_metal/NTT_Metal that come from
    // this allocator hit the `newBufferWithBytesNoCopy` zero-copy path inside
    // the Metal bridge — no memcpy in or memcpy-readback out, which matters
    // for large inputs (e.g., a 262k-row × 128-col tree is 256 MB of input
    // and ~8 MB of tree readback avoided on the `tree` side alone).
    //
    // Returns nullptr on allocation failure. Must be freed with free_aligned.
    Goldilocks::Element* allocate_aligned_elements(uint64_t n);
    void                 free_aligned(void* ptr);
}
#endif
