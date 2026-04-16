// bench_hybrid_batch.mm — tree-count split hybrid Merkle.
//
// Measures merkletree_hybrid_batch (CPU NEON + GPU Metal, split by tree
// count rather than by row) against the Metal-only batched dispatcher.
// Validates bit-exact against per-tree merkletree_neon. Sweeps a few
// cpu_fraction values and reports the auto-calibrated choice.
//
// This is the variant that scales: unlike within-tree hybrid (which hits
// memory-bandwidth contention at multi-GB single trees), batched hybrid
// puts WHOLE independent trees on each engine, so per-engine memory
// traffic stays disjoint.

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

#ifndef MTL_KERNEL_DIR
#error "MTL_KERNEL_DIR must be defined"
#endif

using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

static std::string slurp(const std::string& p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::string strip_includes(const std::string& s) {
    std::stringstream in(s), out; std::string line;
    while (std::getline(in, line)) {
        auto p = line.find_first_not_of(" \t");
        if (p != std::string::npos && line[p] == '#'
            && line.find("include", p) != std::string::npos) continue;
        out << line << "\n";
    }
    return out.str();
}
static void load_lib() {
    MetalCtxHandle ctx = metal_context_get();
    if (metal_context_load_library(ctx, "./goldilocks.metallib") == 0) return;
    std::string dir = MTL_KERNEL_DIR;
    std::string src = strip_includes(slurp(dir + "/field.metal")) +
                      slurp(dir + "/constants.metal.inc") +
                      strip_includes(slurp(dir + "/poseidon.metal"));
    src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + src;
    if (metal_context_load_source(ctx, src.c_str()) != 0) std::abort();
}

static void fib_fill(Goldilocks::Element* cols, uint64_t ncols, uint64_t nrows,
                     uint64_t seed) {
    for (uint64_t i = 0; i < ncols; i++) {
        cols[i]         = Goldilocks::fromU64(seed + i + 1);
        cols[i + ncols] = Goldilocks::fromU64(seed + i + 2);
    }
    for (uint64_t j = 2; j < nrows; j++) {
        for (uint64_t i = 0; i < ncols; i++) {
            cols[j * ncols + i] =
                cols[(j - 2) * ncols + i] + cols[(j - 1) * ncols + i];
        }
    }
}

template <class Fn>
static double time_avg(Fn&& fn, int iters) {
    fn();  // warm-up
    double total = 0.0;
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        total += ms(t1 - t0).count();
    }
    return total / iters;
}

static void bench_shape(uint64_t ncols, uint64_t nrows, uint64_t count, int iters) {
    uint64_t n_elem = ncols * nrows;
    uint64_t n_tree = MerklehashGoldilocks::getTreeNumElements(nrows);

    std::vector<std::vector<Goldilocks::Element>> inputs(count, std::vector<Goldilocks::Element>(n_elem));
    std::vector<std::vector<Goldilocks::Element>> trees_metal(count, std::vector<Goldilocks::Element>(n_tree));
    std::vector<std::vector<Goldilocks::Element>> trees_hyb  (count, std::vector<Goldilocks::Element>(n_tree));
    std::vector<std::vector<Goldilocks::Element>> trees_neon (count, std::vector<Goldilocks::Element>(n_tree));

    for (uint64_t i = 0; i < count; i++) {
        fib_fill(inputs[i].data(), ncols, nrows, i * 1000);
    }

    std::vector<Goldilocks::Element*> in_ptrs(count), m_ptrs(count), h_ptrs(count);
    for (uint64_t i = 0; i < count; i++) {
        in_ptrs[i] = inputs[i].data();
        m_ptrs[i]  = trees_metal[i].data();
        h_ptrs[i]  = trees_hyb[i].data();
    }

    // References
    goldilocks_metal::merkletree_metal_batch(m_ptrs.data(), in_ptrs.data(),
                                               count, ncols, nrows);
    for (uint64_t i = 0; i < count; i++) {
        PoseidonGoldilocks::merkletree_neon(trees_neon[i].data(),
                                             inputs[i].data(), ncols, nrows);
    }
    bool refs_match = true;
    for (uint64_t i = 0; i < count; i++) {
        if (std::memcmp(trees_metal[i].data(), trees_neon[i].data(),
                        n_tree * sizeof(Goldilocks::Element)) != 0) {
            refs_match = false; break;
        }
    }

    printf("\n=== ncols=%llu  nrows=%llu  count=%llu  (iters=%d) ===\n",
           (unsigned long long)ncols,
           (unsigned long long)nrows,
           (unsigned long long)count, iters);
    printf("  ref: metal_batch == per-tree neon: %s\n",
           refs_match ? "match" : "DIVERGE");

    double t_metal_batch = time_avg([&]{
        goldilocks_metal::merkletree_metal_batch(m_ptrs.data(), in_ptrs.data(),
                                                  count, ncols, nrows);
    }, iters);
    printf("  metal_batch       : %9.3f ms (%.3f ms/tree)\n",
           t_metal_batch, t_metal_batch / count);

    double R = goldilocks_metal::get_merkle_throughput_ratio();
    double auto_frac = 1.0 / (1.0 + R);
    printf("  auto-calibration  : R=T_neon/T_metal=%.2f  →  cpu_fraction=%.3f\n",
           R, auto_frac);

    // Try the auto choice + a few neighbors.
    double fracs[] = {auto_frac, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50};
    const int NF = (int)(sizeof(fracs) / sizeof(fracs[0]));

    double best_t   = t_metal_batch;
    double best_frac = 0.0;
    for (int i = 0; i < NF; i++) {
        double t = time_avg([&]{
            PoseidonGoldilocks::merkletree_hybrid_batch(h_ptrs.data(), in_ptrs.data(),
                                                         count, ncols, nrows, fracs[i]);
        }, iters);
        // Bit-exact check
        bool match = true;
        for (uint64_t j = 0; j < count; j++) {
            if (std::memcmp(trees_hyb[j].data(), trees_metal[j].data(),
                            n_tree * sizeof(Goldilocks::Element)) != 0) {
                match = false; break;
            }
        }
        const char* tag = (i == 0) ? " AUTO" : "     ";
        printf("    cpu_frac=%.3f%s : %9.3f ms (%5.2fx vs metal_batch, match=%s)\n",
               fracs[i], tag, t, t_metal_batch / t, match ? "yes" : "NO");
        if (t < best_t && match) { best_t = t; best_frac = fracs[i]; }
    }
    printf("  best hybrid       : %9.3f ms at cpu_fraction=%.3f  (%5.2fx vs metal_batch)\n",
           best_t, best_frac, t_metal_batch / best_t);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        load_lib();
        printf("bench_hybrid_batch: tree-count CPU NEON + GPU Metal split\n");
        bool big = (argc > 1 && std::string(argv[1]) == "--big");

        // Realistic prover patterns: many medium-sized trees.
        bench_shape(128,   65536,  4, 3);   //   4 trees × 64k = ~32 MB each
        bench_shape(128,   65536, 16, 2);   //  16 trees × 64k
        bench_shape(128,  262144,  4, 2);   //   4 trees × 256k
        bench_shape(128,  262144,  8, 2);   //   8 trees × 256k
        bench_shape(128, 1048576,  4, 1);   //   4 trees × 1M
        if (big) {
            bench_shape(128, 1048576,  8, 1);  // 8 GB total, ~6 GB on GPU
            bench_shape(128, 1ULL << 23, 2, 1); // 2 trees × 8M = MERKLETREE_BENCH × 2
        }
    }
    return 0;
}
