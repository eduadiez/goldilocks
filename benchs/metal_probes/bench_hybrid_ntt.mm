// bench_hybrid_ntt.mm — measure memcpy-based hybrid CPU NEON + GPU Metal NTT.
//
// NTT can only be split by COLUMNS (rows can't be split — the butterflies
// at later phases couple every row). The Metal NTT kernel uses `ncols` as
// both the column count AND the row stride, so we can't process a "slice"
// of cols without either:
//   (a) modifying the kernel to take stride and ncols_proc separately, or
//   (b) extracting a contiguous (size × K_gpu) sub-matrix via memcpy,
//       running NTT on that, and reassembling back into the interleaved
//       layout via memcpy.
//
// This probe implements (b) and measures whether the memcpy round-trip
// overhead leaves any net win. Spoiler: at production scale it doesn't —
// the conclusion drives whether (a) is worth the engineering effort of
// rewriting every NTT kernel.
//
// For each shape we report:
//   T_metal_only       — pure GPU NTT
//   T_neon_only        — pure CPU NEON NTT
//   T_split_in         — memcpy interleaved → contiguous GPU + CPU buffers
//   T_split_out        — memcpy contiguous buffers back to interleaved
//   T_hybrid_compute   — max(GPU NTT on its half, CPU NTT on its half)
//   T_hybrid_total     — T_split_in + T_hybrid_compute + T_split_out
//   speedup            — T_metal_only / T_hybrid_total

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

// Extract cols [col_start, col_start + col_count) from interleaved input
// (laid out as buf[row * ncols + col]) into a contiguous output sub-matrix
// of shape (size, col_count) with stride col_count.
static void split_columns(Goldilocks::Element* dst,
                          const Goldilocks::Element* src,
                          uint64_t size, uint64_t ncols,
                          uint64_t col_start, uint64_t col_count) {
    if (col_count == ncols) {
        // Whole-matrix copy — single memcpy, fastest path.
        std::memcpy(dst, src, size * ncols * sizeof(Goldilocks::Element));
        return;
    }
    #pragma omp parallel for schedule(static)
    for (uint64_t r = 0; r < size; r++) {
        std::memcpy(&dst[r * col_count],
                    &src[r * ncols + col_start],
                    col_count * sizeof(Goldilocks::Element));
    }
}
// Reassemble cols [col_start, col_start + col_count) from a contiguous
// sub-matrix back into the interleaved layout.
static void merge_columns(Goldilocks::Element* dst, uint64_t ncols,
                          uint64_t col_start, uint64_t col_count,
                          const Goldilocks::Element* src,
                          uint64_t size) {
    if (col_count == ncols) {
        std::memcpy(dst, src, size * ncols * sizeof(Goldilocks::Element));
        return;
    }
    #pragma omp parallel for schedule(static)
    for (uint64_t r = 0; r < size; r++) {
        std::memcpy(&dst[r * ncols + col_start],
                    &src[r * col_count],
                    col_count * sizeof(Goldilocks::Element));
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

static void bench_shape(uint64_t N, uint64_t ncols, int iters) {
    uint64_t total = N * ncols;
    std::vector<Goldilocks::Element> input(total), output(total);
    fill_seq(input.data(), total);

    NTT_Goldilocks ntt(N);

    double t_metal = time_avg([&]{
        ntt.NTT_Metal(output.data(), input.data(), N, ncols, false);
    }, iters);
    double t_neon = time_avg([&]{
        ntt.NTT(output.data(), input.data(), N, ncols);
    }, iters);

    // Choose the optimal cpu_fraction from the measured ratio:
    //   cpu_fraction = T_metal / (T_metal + T_neon)
    double cpu_fraction = t_metal / (t_metal + t_neon);
    if (cpu_fraction < 0.05) cpu_fraction = 0.05;
    if (cpu_fraction > 0.95) cpu_fraction = 0.95;
    uint64_t K_cpu = (uint64_t)(ncols * cpu_fraction);
    if (K_cpu == 0) K_cpu = 1;
    if (K_cpu >= ncols) K_cpu = ncols - 1;
    uint64_t K_gpu = ncols - K_cpu;

    // Per-row cost varies: lots of memory pressure if the workload is large.
    // We isolate split / compute / merge so the user can see where time goes.
    std::vector<Goldilocks::Element> gpu_in (N * K_gpu), gpu_out(N * K_gpu);
    std::vector<Goldilocks::Element> cpu_in (N * K_cpu), cpu_out(N * K_cpu);

    double t_split = time_avg([&]{
        split_columns(gpu_in.data(), input.data(), N, ncols, 0,     K_gpu);
        split_columns(cpu_in.data(), input.data(), N, ncols, K_gpu, K_cpu);
    }, iters);

    double t_compute = time_avg([&]{
        std::thread metal_thread([&]{
            ntt.NTT_Metal(gpu_out.data(), gpu_in.data(), N, K_gpu, false);
        });
        ntt.NTT(cpu_out.data(), cpu_in.data(), N, K_cpu);
        metal_thread.join();
    }, iters);

    double t_merge = time_avg([&]{
        merge_columns(output.data(), ncols, 0,     K_gpu, gpu_out.data(), N);
        merge_columns(output.data(), ncols, K_gpu, K_cpu, cpu_out.data(), N);
    }, iters);

    double t_hybrid_total = t_split + t_compute + t_merge;
    double speedup = t_metal / t_hybrid_total;

    // Bit-exact check: rerun once and compare to the metal-only output.
    std::vector<Goldilocks::Element> ref(total);
    ntt.NTT_Metal(ref.data(), input.data(), N, ncols, false);

    split_columns(gpu_in.data(), input.data(), N, ncols, 0,     K_gpu);
    split_columns(cpu_in.data(), input.data(), N, ncols, K_gpu, K_cpu);
    {
        std::thread metal_thread([&]{
            ntt.NTT_Metal(gpu_out.data(), gpu_in.data(), N, K_gpu, false);
        });
        ntt.NTT(cpu_out.data(), cpu_in.data(), N, K_cpu);
        metal_thread.join();
    }
    merge_columns(output.data(), ncols, 0,     K_gpu, gpu_out.data(), N);
    merge_columns(output.data(), ncols, K_gpu, K_cpu, cpu_out.data(), N);
    bool match = (std::memcmp(output.data(), ref.data(),
                              total * sizeof(Goldilocks::Element)) == 0);

    printf("\n=== NTT  N=2^%d (%llu)  ncols=%llu  (iters=%d) ===\n",
           (int)__builtin_ctzll(N),
           (unsigned long long)N, (unsigned long long)ncols, iters);
    printf("  metal only      : %9.3f ms\n", t_metal);
    printf("  neon  only      : %9.3f ms   (ratio metal:neon = 1 : %.2f)\n",
           t_neon, t_neon / t_metal);
    printf("  picked split    : K_gpu=%llu (%.0f%%)  K_cpu=%llu (%.0f%%)\n",
           (unsigned long long)K_gpu, 100.0 * K_gpu / ncols,
           (unsigned long long)K_cpu, 100.0 * K_cpu / ncols);
    printf("  split (memcpy)  : %9.3f ms\n", t_split);
    printf("  compute (concur): %9.3f ms\n", t_compute);
    printf("  merge (memcpy)  : %9.3f ms\n", t_merge);
    printf("  hybrid total    : %9.3f ms   (%5.2fx vs metal-only, match=%s)\n",
           t_hybrid_total, speedup, match ? "yes" : "NO");
    if (speedup >= 1.05) {
        printf("  verdict         : WIN — hybrid faster than metal-only\n");
    } else if (speedup >= 0.95) {
        printf("  verdict         : NEUTRAL — within noise of metal-only\n");
    } else {
        printf("  verdict         : LOSE — memcpy overhead beats hybrid gain\n");
    }
}

int main(int argc, char** argv) {
    @autoreleasepool {
        load_lib();
        printf("bench_hybrid_ntt: memcpy-based CPU NEON + GPU Metal NTT (column-split)\n");
        bool big = (argc > 1 && std::string(argv[1]) == "--big");

        bench_shape(1ULL << 16,  64, 5);
        bench_shape(1ULL << 18,  64, 3);
        bench_shape(1ULL << 18, 128, 2);
        bench_shape(1ULL << 16, 256, 3);
        if (big) {
            bench_shape(1ULL << 20, 100, 2);
            bench_shape(1ULL << 23, 100, 1);  // NTT_BENCH shape (~6 GB)
        }
    }
    return 0;
}
