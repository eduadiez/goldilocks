// bench_hybrid_ntt_v2.mm — zero-copy hybrid CPU NEON + GPU Metal NTT.
//
// Uses goldilocks_metal::NTT_Metal_partial (which calls the new
// stride-aware NTT kernels) so the GPU operates on a contiguous column
// SLICE of the original interleaved buffer — no memcpy split/reassemble
// like bench_hybrid_ntt's memcpy-based approach.
//
// However: the CPU side (NEON NTT_Goldilocks::NTT) still expects a
// CONTIGUOUS (size × ncols_cpu) sub-matrix and treats `ncols_cpu` as
// both stride and width. So for the CPU half we still need to extract
// its columns, which means memcpy is unavoidable on that side until the
// CPU NTT also gains stride support.
//
// This probe measures TWO scenarios:
//   1. GPU-only on the full matrix (baseline).
//   2. Hybrid: GPU on cols [0, K_gpu) of the original buffer (zero-copy
//      via NTT_Metal_partial), CPU on cols [K_gpu, ncols) extracted to
//      a temp buffer (memcpy in/out).
//
// At small/medium shapes the CPU-side memcpy is cheap → hybrid wins.
// At LDE_BENCH scale the memcpy still costs but only on the CPU's share
// (~30% of the columns), so it's much smaller than the v1 full-buffer
// memcpy.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <thread>
#include <string>
#include <fstream>
#include <sstream>

#include "../../src/platform.hpp"
#include "../../src/goldilocks_base_field.hpp"
#include "../../src/ntt_goldilocks.hpp"
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
                      strip_includes(slurp(dir + "/ntt.metal"));
    src = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + src;
    if (metal_context_load_source(ctx, src.c_str()) != 0) std::abort();
}

static void fill_seq(Goldilocks::Element* buf, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) buf[i] = Goldilocks::fromU64(i + 1);
}

template <class Fn>
static double time_avg(Fn&& fn, int iters) {
    fn();
    double total = 0.0;
    for (int i = 0; i < iters; i++) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        total += ms(t1 - t0).count();
    }
    return total / iters;
}

// Extract / merge for the CPU half (still memcpy because NEON NTT lacks
// stride support — comparable to bench_hybrid_ntt's memcpy_split, but
// only on the CPU's column share rather than the whole matrix).
static void split_cols(Goldilocks::Element* dst,
                       const Goldilocks::Element* src,
                       uint64_t size, uint64_t ncols_total,
                       uint64_t col_start, uint64_t col_count) {
    #pragma omp parallel for schedule(static)
    for (uint64_t r = 0; r < size; r++) {
        std::memcpy(&dst[r * col_count],
                    &src[r * ncols_total + col_start],
                    col_count * sizeof(Goldilocks::Element));
    }
}
static void merge_cols(Goldilocks::Element* dst, uint64_t ncols_total,
                       uint64_t col_start, uint64_t col_count,
                       const Goldilocks::Element* src,
                       uint64_t size) {
    #pragma omp parallel for schedule(static)
    for (uint64_t r = 0; r < size; r++) {
        std::memcpy(&dst[r * ncols_total + col_start],
                    &src[r * col_count],
                    col_count * sizeof(Goldilocks::Element));
    }
}

static void bench_shape(uint64_t N, uint64_t ncols, int iters) {
    uint64_t total = N * ncols;
    std::vector<Goldilocks::Element> input(total);
    std::vector<Goldilocks::Element> ref_out(total);
    std::vector<Goldilocks::Element> hyb_out(total);
    fill_seq(input.data(), total);

    NTT_Goldilocks ntt(N);

    // Baseline: full GPU NTT (no hybrid).
    double t_metal = time_avg([&]{
        ntt.NTT_Metal(ref_out.data(), input.data(), N, ncols, false);
    }, iters);

    // Choose K_gpu by GPU/CPU throughput ratio. We measure pure NEON for
    // calibration; in production this ratio could come from
    // get_merkle_throughput_ratio's NTT analog.
    double t_neon = time_avg([&]{
        ntt.NTT(ref_out.data(), input.data(), N, ncols);
    }, iters);
    double cpu_fraction = t_metal / (t_metal + t_neon);
    if (cpu_fraction < 0.05) cpu_fraction = 0.05;
    if (cpu_fraction > 0.95) cpu_fraction = 0.95;
    uint64_t K_cpu = (uint64_t)(ncols * cpu_fraction);
    if (K_cpu == 0) K_cpu = 1;
    if (K_cpu >= ncols) K_cpu = ncols - 1;
    uint64_t K_gpu = ncols - K_cpu;

    // Hybrid: GPU on cols [0, K_gpu) of input/hyb_out (zero-copy via stride),
    // CPU on cols [K_gpu, ncols) via extracted sub-matrix.
    std::vector<Goldilocks::Element> cpu_in(N * K_cpu), cpu_out(N * K_cpu);

    double t_hybrid = time_avg([&]{
        // Set up CPU input slice (memcpy out of original).
        split_cols(cpu_in.data(), input.data(), N, ncols, K_gpu, K_cpu);

        // Spawn GPU thread: stride-aware NTT_Metal_partial on cols [0, K_gpu).
        std::thread gpu_thread([&]{
            goldilocks_metal::NTT_Metal_partial(
                hyb_out.data(),       // dst slice base = output's col 0
                input.data(),         // src slice base = input's col 0
                N,
                /*ncols_proc=*/K_gpu,
                /*ncols_stride=*/ncols,
                &ntt,
                /*inverse=*/false);
        });
        // CPU: NEON NTT on the extracted sub-matrix.
        ntt.NTT(cpu_out.data(), cpu_in.data(), N, K_cpu);
        gpu_thread.join();

        // Merge CPU output back into the interleaved layout.
        merge_cols(hyb_out.data(), ncols, K_gpu, K_cpu, cpu_out.data(), N);
    }, iters);

    bool match = (std::memcmp(hyb_out.data(), ref_out.data(),
                              total * sizeof(Goldilocks::Element)) == 0);

    printf("\n=== NTT  N=2^%d (%llu)  ncols=%llu  (iters=%d) ===\n",
           (int)__builtin_ctzll(N),
           (unsigned long long)N, (unsigned long long)ncols, iters);
    printf("  metal only       : %9.3f ms\n", t_metal);
    printf("  neon  only       : %9.3f ms   (ratio = 1 : %.2f)\n",
           t_neon, t_neon / t_metal);
    printf("  picked split     : K_gpu=%llu (%.0f%%)  K_cpu=%llu (%.0f%%)\n",
           (unsigned long long)K_gpu, 100.0 * K_gpu / ncols,
           (unsigned long long)K_cpu, 100.0 * K_cpu / ncols);
    printf("  hybrid (zero-copy GPU)  : %9.3f ms   (%5.2fx vs metal-only, match=%s)\n",
           t_hybrid, t_metal / t_hybrid, match ? "yes" : "NO");
}

int main(int argc, char** argv) {
    @autoreleasepool {
        load_lib();
        printf("bench_hybrid_ntt_v2: zero-copy CPU+GPU column split via NTT_Metal_partial\n");

        bool big = (argc > 1 && std::string(argv[1]) == "--big");

        bench_shape(1ULL << 16,  64, 5);
        bench_shape(1ULL << 18,  64, 3);
        bench_shape(1ULL << 18, 128, 2);
        bench_shape(1ULL << 16, 256, 3);
        if (big) {
            bench_shape(1ULL << 20, 100, 2);
            bench_shape(1ULL << 23, 100, 1);  // NTT_BENCH shape
        }
    }
    return 0;
}
