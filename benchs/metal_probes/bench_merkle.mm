// bench_merkle.mm — Metal vs NEON vs scalar Merkle tree throughput.
//
// Runs merkletree_{metal,neon,seq} on a fixed Fibonacci-2D input at three
// sizes. Reports mean ms/iter, with a warm-up iteration discarded. Uses the
// actual library entry points so results match what callers would see.
//
// Build (from this directory):
//   make bench_merkle
// Run:
//   ./bench_merkle
//
// The MSL kernels are loaded at runtime via metal_context_load_source so a
// pre-built .metallib is not required.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "../../src/platform.hpp"
#include "../../src/goldilocks_base_field.hpp"
#include "../../src/poseidon_goldilocks.hpp"
#include "../../src/merklehash_goldilocks.hpp"
#include "../../src/metal/metal_context.hpp"
#include "../../src/metal/goldilocks_metal.hpp"

// Opt-in switch: when set, merkletree_metal dispatches the SIMD-cooperative
// leaf kernel instead of the one-thread-per-row kernel. Read from env var
// GOLDILOCKS_METAL_COOP=1 by main().
extern "C" { int g_merkle_use_simd_coop = 0; }

#ifndef MTL_KERNEL_DIR
#error "MTL_KERNEL_DIR must be defined (path to src/metal/kernels)"
#endif

using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string strip_includes(const std::string& src) {
    std::stringstream in(src);
    std::stringstream out;
    std::string line;
    while (std::getline(in, line)) {
        auto p = line.find_first_not_of(" \t");
        if (p != std::string::npos && line[p] == '#') {
            auto q = line.find("include", p);
            if (q != std::string::npos) continue;
        }
        out << line << "\n";
    }
    return out.str();
}

static void load_metallib_or_source() {
    MetalCtxHandle ctx = metal_context_get();
    if (metal_context_load_library(ctx, "./goldilocks.metallib") == 0) return;
    std::string dir = MTL_KERNEL_DIR;
    std::string src = strip_includes(slurp(dir + "/field.metal")) +
                      slurp(dir + "/constants.metal.inc") +
                      strip_includes(slurp(dir + "/poseidon.metal"));
    // Prepend a stdlib include since all strip_includes removed them.
    src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + src;
    if (metal_context_load_source(ctx, src.c_str()) != 0) {
        fprintf(stderr, "bench_merkle: runtime source compile failed\n");
        std::abort();
    }
}

static void build_fibonacci(Goldilocks::Element* cols,
                            uint64_t ncols, uint64_t nrows) {
    for (uint64_t i = 0; i < ncols; i++) {
        cols[i]            = Goldilocks::fromU64(i) + Goldilocks::one();
        cols[i + ncols]    = Goldilocks::fromU64(i) + Goldilocks::one();
    }
    for (uint64_t j = 2; j < nrows; j++) {
        for (uint64_t i = 0; i < ncols; i++) {
            cols[j * ncols + i] =
                cols[(j - 2) * ncols + i] + cols[(j - 1) * ncols + i];
        }
    }
}

struct Result { double ms; uint64_t root0; };

template <class Fn>
static Result time_run(Fn&& fn,
                       Goldilocks::Element* tree,
                       uint64_t numTreeElems,
                       int iters) {
    // Warm-up
    fn();
    uint64_t r0 = Goldilocks::toU64(tree[numTreeElems - 4]);

    double total = 0.0;
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        total += ms(t1 - t0).count();
    }
    return { total / iters, r0 };
}

static void bench_one_size(uint64_t ncols, uint64_t nrows, int iters) {
    uint64_t n_elem = ncols * nrows;
    uint64_t n_tree = MerklehashGoldilocks::getTreeNumElements(nrows);
    auto* cols = new Goldilocks::Element[n_elem];
    auto* tree = new Goldilocks::Element[n_tree];

    build_fibonacci(cols, ncols, nrows);

    Result r_seq = time_run(
        [&]{ PoseidonGoldilocks::merkletree_seq(tree, cols, ncols, nrows); },
        tree, n_tree, iters);

#ifdef GOLDILOCKS_HAS_NEON
    Result r_neon = time_run(
        [&]{ PoseidonGoldilocks::merkletree_neon(tree, cols, ncols, nrows); },
        tree, n_tree, iters);
#else
    Result r_neon = { 0.0, 0 };
#endif

    // Metal, one-thread-per-row kernel (default)
    g_merkle_use_simd_coop = 0;
    Result r_metal = time_run(
        [&]{ PoseidonGoldilocks::merkletree_metal(tree, cols, ncols, nrows); },
        tree, n_tree, iters);

    // Metal, SIMD-group-cooperative kernel
    g_merkle_use_simd_coop = 1;
    Result r_coop = time_run(
        [&]{ PoseidonGoldilocks::merkletree_metal(tree, cols, ncols, nrows); },
        tree, n_tree, iters);

    // Metal, x2 ILP kernel (2 rows per thread)
    g_merkle_use_simd_coop = 2;
    Result r_x2 = time_run(
        [&]{ PoseidonGoldilocks::merkletree_metal(tree, cols, ncols, nrows); },
        tree, n_tree, iters);

    // Metal, column-major / coalesced reads
    g_merkle_use_simd_coop = 3;
    Result r_cm = time_run(
        [&]{ PoseidonGoldilocks::merkletree_metal(tree, cols, ncols, nrows); },
        tree, n_tree, iters);

    // Metal, fused-tile cooperative load
    g_merkle_use_simd_coop = 4;
    Result r_tg = time_run(
        [&]{ PoseidonGoldilocks::merkletree_metal(tree, cols, ncols, nrows); },
        tree, n_tree, iters);
    g_merkle_use_simd_coop = 0;

    // Metal, row kernel with PAGE-ALIGNED buffers (zero-copy MTLBuffer alias)
    auto* cols_al = goldilocks_metal::allocate_aligned_elements(n_elem);
    auto* tree_al = goldilocks_metal::allocate_aligned_elements(n_tree);
    build_fibonacci(cols_al, ncols, nrows);
    Result r_alig = time_run(
        [&]{ PoseidonGoldilocks::merkletree_metal(tree_al, cols_al, ncols, nrows); },
        tree_al, n_tree, iters);
    goldilocks_metal::free_aligned(cols_al);
    goldilocks_metal::free_aligned(tree_al);

    const char* check_seq_neon  = (r_seq.root0 == r_neon.root0)  ? "match" : "DIVERGE";
    const char* check_seq_metal = (r_seq.root0 == r_metal.root0) ? "match" : "DIVERGE";
    const char* check_seq_coop  = (r_seq.root0 == r_coop.root0)  ? "match" : "DIVERGE";
    const char* check_seq_x2    = (r_seq.root0 == r_x2.root0)    ? "match" : "DIVERGE";
    const char* check_seq_cm    = (r_seq.root0 == r_cm.root0)    ? "match" : "DIVERGE";
    const char* check_seq_tg    = (r_seq.root0 == r_tg.root0)    ? "match" : "DIVERGE";
    const char* check_seq_alig  = (r_seq.root0 == r_alig.root0)  ? "match" : "DIVERGE";

    printf("\n=== Merkle  ncols=%llu  nrows=%llu  (iters=%d) ===\n",
           (unsigned long long)ncols, (unsigned long long)nrows, iters);
    printf("  seq          : %10.3f ms\n", r_seq.ms);
#ifdef GOLDILOCKS_HAS_NEON
    printf("  neon         : %10.3f ms   (%5.2fx vs seq,  check=%s)\n",
           r_neon.ms, r_seq.ms / r_neon.ms, check_seq_neon);
#else
    printf("  neon         : n/a (GOLDILOCKS_HAS_NEON not defined)\n");
#endif
    printf("  metal (row)  : %10.3f ms   (%5.2fx vs neon, check=%s)\n",
           r_metal.ms,
           r_neon.ms > 0 ? r_neon.ms / r_metal.ms : 0.0,
           check_seq_metal);
    printf("  metal (coop) : %10.3f ms   (%5.2fx vs neon, %5.2fx vs metal-row, check=%s)\n",
           r_coop.ms,
           r_neon.ms  > 0 ? r_neon.ms  / r_coop.ms : 0.0,
           r_metal.ms > 0 ? r_metal.ms / r_coop.ms : 0.0,
           check_seq_coop);
    printf("  metal (x2)   : %10.3f ms   (%5.2fx vs neon, %5.2fx vs metal-row, check=%s)\n",
           r_x2.ms,
           r_neon.ms  > 0 ? r_neon.ms  / r_x2.ms : 0.0,
           r_metal.ms > 0 ? r_metal.ms / r_x2.ms : 0.0,
           check_seq_x2);
    printf("  metal (cm)   : %10.3f ms   (%5.2fx vs neon, %5.2fx vs metal-row, check=%s)\n",
           r_cm.ms,
           r_neon.ms  > 0 ? r_neon.ms  / r_cm.ms : 0.0,
           r_metal.ms > 0 ? r_metal.ms / r_cm.ms : 0.0,
           check_seq_cm);
    printf("  metal (tg)   : %10.3f ms   (%5.2fx vs neon, %5.2fx vs metal-row, check=%s)\n",
           r_tg.ms,
           r_neon.ms  > 0 ? r_neon.ms  / r_tg.ms : 0.0,
           r_metal.ms > 0 ? r_metal.ms / r_tg.ms : 0.0,
           check_seq_tg);
    printf("  metal+aligned: %10.3f ms   (%5.2fx vs neon, %5.2fx vs metal-row, check=%s)\n",
           r_alig.ms,
           r_neon.ms  > 0 ? r_neon.ms  / r_alig.ms : 0.0,
           r_metal.ms > 0 ? r_metal.ms / r_alig.ms : 0.0,
           check_seq_alig);

    delete[] cols;
    delete[] tree;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        load_metallib_or_source();
        printf("bench_merkle: Apple M-series Goldilocks Merkle (seq / neon / metal)\n");

        bool big = (argc > 1 && std::string(argv[1]) == "--big");

        // Small: oracle size (primarily correctness + dispatch overhead)
        bench_one_size(128,   64,     50);
        bench_one_size(128,   4096,   20);
        bench_one_size(128,   65536,   5);
        bench_one_size(128,  262144,   3);

        if (big) {
            printf("\n--- production-scale shapes (larger ncols / nrows) ---\n");
            // Realistic STARK prover shapes:
            //   FFT-domain merkle at 2^18..2^20 with 64-256 cols.
            //   Input sizes here:  1 GiB .. 2 GiB per input.
            bench_one_size(64,    (uint64_t)(1ULL << 20),   2);  //  64 cols × 1M rows   = 512 MB
            bench_one_size(128,   (uint64_t)(1ULL << 20),   2);  // 128 cols × 1M rows   = 1 GB
            bench_one_size(256,   (uint64_t)(1ULL << 18),   2);  // 256 cols × 256k rows = 512 MB
            bench_one_size(512,   (uint64_t)(1ULL << 18),   2);  // 512 cols × 256k rows = 1 GB
        }
    }
    return 0;
}
