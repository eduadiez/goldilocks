// bench_hybrid_merkle.mm — CPU NEON + GPU Metal concurrent Merkle build.
//
// Validates `PoseidonGoldilocks::merkletree_hybrid` output against both the
// Metal-only and NEON-only reference builds (bit-exact via memcmp on the
// root). Then sweeps a range of cpu_fraction values to find the optimum
// split ratio empirically and compares to Metal-only throughput at the
// best setting.
//
// Build: make bench_hybrid_merkle

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

static void fib_fill(Goldilocks::Element* cols, uint64_t ncols, uint64_t nrows) {
    for (uint64_t i = 0; i < ncols; i++) {
        cols[i]         = Goldilocks::fromU64(i + 1);
        cols[i + ncols] = Goldilocks::fromU64(i + 2);
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

static void bench_shape(uint64_t ncols, uint64_t nrows, int iters) {
    uint64_t n_elem = ncols * nrows;
    uint64_t n_tree = MerklehashGoldilocks::getTreeNumElements(nrows);

    std::vector<Goldilocks::Element> cols(n_elem);
    std::vector<Goldilocks::Element> tree_metal(n_tree);
    std::vector<Goldilocks::Element> tree_neon (n_tree);
    std::vector<Goldilocks::Element> tree_hyb  (n_tree);
    fib_fill(cols.data(), ncols, nrows);

    // References
    PoseidonGoldilocks::merkletree_metal(tree_metal.data(), cols.data(), ncols, nrows);
    PoseidonGoldilocks::merkletree_neon (tree_neon.data(),  cols.data(), ncols, nrows);
    bool metal_eq_neon = (std::memcmp(tree_metal.data(), tree_neon.data(),
                                      n_tree * sizeof(Goldilocks::Element)) == 0);

    printf("\n=== ncols=%llu  nrows=%llu  (iters=%d) ===\n",
           (unsigned long long)ncols, (unsigned long long)nrows, iters);
    printf("  reference: metal == neon: %s\n",
           metal_eq_neon ? "match" : "DIVERGE");

    // Baseline timings
    double t_metal = time_avg([&]{
        PoseidonGoldilocks::merkletree_metal(tree_metal.data(), cols.data(), ncols, nrows);
    }, iters);
    double t_neon = time_avg([&]{
        PoseidonGoldilocks::merkletree_neon(tree_neon.data(), cols.data(), ncols, nrows);
    }, iters);
    printf("  metal only  : %9.3f ms\n", t_metal);
    printf("  neon  only  : %9.3f ms\n", t_neon);

    // Sweep cpu_fraction. We expect a valley around cpu_fraction ≈ t_gpu /
    // (t_gpu + t_cpu) — that's the balanced-finish point.
    double cpu_fracs[] = {0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50};
    const int NF = (int)(sizeof(cpu_fracs) / sizeof(cpu_fracs[0]));
    double best_t   = t_metal;
    double best_cpu = 0.0;

    printf("  hybrid sweep:\n");
    for (int i = 0; i < NF; i++) {
        double t = time_avg([&]{
            PoseidonGoldilocks::merkletree_hybrid(tree_hyb.data(), cols.data(),
                                                   ncols, nrows, cpu_fracs[i]);
        }, iters);
        bool match = (std::memcmp(tree_hyb.data(), tree_metal.data(),
                                  n_tree * sizeof(Goldilocks::Element)) == 0);
        printf("    cpu_fraction=%.2f : %9.3f ms   (%5.2fx vs metal, match=%s)\n",
               cpu_fracs[i], t, t_metal / t, match ? "yes" : "NO");
        if (t < best_t && match) { best_t = t; best_cpu = cpu_fracs[i]; }
    }
    printf("  best hybrid : %9.3f ms  at cpu_fraction=%.2f  (%5.2fx vs metal-only)\n",
           best_t, best_cpu, t_metal / best_t);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        load_lib();
        printf("bench_hybrid_merkle: CPU NEON + GPU Metal concurrent Merkle\n");
        bool big = (argc > 1 && std::string(argv[1]) == "--big");

        bench_shape(128,   4096,   15);
        bench_shape(128,  65536,    5);
        bench_shape(128, 262144,    3);
        if (big) {
            bench_shape(128, 1ULL << 20,  2);
            bench_shape(128, 1ULL << 23,  1);
        }
    }
    return 0;
}
