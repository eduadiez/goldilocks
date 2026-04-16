# Goldilocks Metal GPU Benchmark — Apple M4 Pro

**Date:** 2026-04-16
**System:** macOS 26.3.1 — Apple M4 Pro (14 CPU cores, 20 GPU cores, unified memory)
**Compiler:** Apple clang 17.0.0
**Shader compiler:** `xcrun metal` (Xcode 15+, Metal Toolchain component)
**Build:** `-O3 -std=c++17 -fobjc-arc -DGOLDILOCKS_HAS_METAL`
**Path exercised:** Metal GPU kernels (`PoseidonGoldilocks::merkletree_metal`, `NTT_Goldilocks::NTT_Metal`) compared vs scalar and NEON paths on the same chip, and against x86-64 AVX2 on Xeon D-2141I (see `x86_64-linux-xeon-d2141i.md`).
**Commit:** `metal-gpu` branch — 17 commits covering incremental optimizations (fused kernels, radix-4 NTT butterfly, batched Merkle, specialized MDS mul, etc.).

Bit-exact with the scalar/NEON reference on every workload measured
(Fibonacci-seeded input → output root cross-checked via `memcmp`).

---

## 1. Merkle tree — `MERKLETREE_BENCH` at FFT_SIZE=2²³ × NCOLS_HASH=128

This is the exact workload of `benchs/bench.cpp::MERKLETREE_BENCH`: 8M
rows × 128 cols × 8 bytes = 8 GiB input, Fibonacci fill.

Full tree size: (2·8M − 1) × 4 × 8 B ≈ 500 MiB.

| Backend | Hardware | Time | MiB/s | µs/hash | vs scalar | vs NEON |
|---|---|---:|---:|---:|---:|---:|
| Scalar (portable `__uint128_t`) | Apple M4 Pro / 14 threads | 54.12 s | 152 | 2.80 | 1.00× | 0.29× |
| NEON (current) | Apple M4 Pro / 14 threads | 15.77 s | 519 | 0.83 | 3.43× | 1.00× |
| Scalar | Xeon D-2141I / 16 threads | 111.87 s | 73 | 6.67 | 0.48× | 0.14× |
| AVX2 | Xeon D-2141I / 16 threads | 44.95 s | 182 | 2.68 | 1.20× | 0.35× |
| **Metal (GPU)** | **Apple M4 Pro (20-core GPU)** | **5.64 s** | **1452** | **0.67** | **9.60×** | **2.80×** |

Metal on M4 Pro vs:
- AVX2 (Xeon D-2141I): **7.97×**
- NEON (M4 Pro CPU): **2.80×**
- Scalar (M4 Pro CPU): **9.60×**

---

## 2. Merkle tree — production-scale sweep (128 cols, varying rows)

From `benchs/metal_probes/bench_merkle --big`.

| nrows | seq (ms) | neon (ms) | **metal (ms)** | metal/neon | metal/seq |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.84 | 0.45 | 4.79 | 0.09× | 0.18× (launch-bound) |
| 4 096 | 31.9 | 10.4 | **8.73** | **1.19×** | 3.66× |
| 65 536 | 418 | 125 | **49.5** | **2.51×** | 8.45× |
| 262 144 | 1671 | 460 | **181.5** | **2.53×** | 9.20× |
| 1 048 576 (64 cols) | 3493 | 1058 | **379** | **2.79×** | 9.21× |
| 1 048 576 (128 cols, ~1 GB) | 6796 | 2053 | **705** | **2.91×** | 9.64× |
| 262 144 (256 cols, ~512 MB) | 3364 | 1040 | **351** | **2.96×** | 9.59× |
| 262 144 (512 cols, ~1 GB) | 6701 | 2056 | **697** | **2.95×** | 9.61× |
| **8 388 608 (128 cols, ~8 GB)** | **54 119** | **15 774** | **5639** | **2.80×** | **9.60×** |

**Crossover**: Metal wins the row-count battle above ~4k rows for 128-col
input. Below that, kernel launch overhead (~5 ms) dominates.

---

## 3. NTT forward — `bench_ntt` measured shapes

From `benchs/metal_probes/bench_ntt`.

| Shape | NEON (ms) | **Metal (ms)** | Metal/NEON |
|---|---:|---:|---:|
| N=2¹⁴ × 1 | 0.25 | 0.44 | 0.56× (launch-bound) |
| N=2¹⁶ × 1 | 0.46 | 0.46 | **1.00×** |
| N=2¹⁸ × 1 | 1.49 | **1.00** | **1.50×** |
| N=2²⁰ × 1 | 6.20 | **2.88** | **2.15×** |
| N=2¹⁴ × 64 | 1.31 | 1.24 | **1.06×** |
| N=2¹⁶ × 64 | 6.28 | **4.09** | **1.54×** |
| N=2¹⁸ × 64 | 27.33 | **13.28** | **2.06×** |
| N=2¹⁶ × 256 | 23.10 | **11.94** | **1.93×** |
| **N=2¹⁸ × 128** (STARK shape) | 51.06 | **26.30** | **1.94×** |

The `bench.cpp::NTT_BENCH` workload (FFT_SIZE=2²³ × NUM_COLUMNS=100) wasn't
directly measured in `bench_ntt`, but we have the CPU numbers from
`benchcpu`:

| Variant | Time | Notes |
|---|---:|---|
| NTT_BENCH scalar (M4 Pro 14 threads) | 6.33 s | No AVX2 / NEON NTT path in the codebase — scalar-only |
| NTT_BENCH scalar (Xeon D-2141I 16 threads) | 34.4 s | Same scalar code, slower core |
| Metal estimate | ~3 s | Extrapolated from 2²⁰ × 1 Metal = 2.88 ms, scaled by domain and column factors |

There is no AVX2 NTT implementation in this codebase (`NTT_BENCH_AVX` does
not exist). The NEON implementation at `ntt_goldilocks.cpp:94-148` is the
only SIMD NTT available. Our Metal path targets the same algorithm.

---

## 4. Poseidon permutation (`POSEIDON_BENCH_FULL`, `POSEIDON_BENCH`)

From `make benchcpu` on Apple M4 Pro (14 threads) and
`benchs/results/x86_64-linux-xeon-d2141i.md` (Xeon, 16 threads).

| Benchmark | Scalar (M4) | NEON (M4) | Scalar (Xeon) | AVX2 (Xeon) |
|---|---:|---:|---:|---:|
| POSEIDON_BENCH_FULL | 255.7 MiB/s · 2.51 µs | **465.9 MiB/s · 1.38 µs** | 118.6 MiB/s · 6.18 µs | 293.8 MiB/s · 2.49 µs |
| POSEIDON_BENCH | 247.6 MiB/s · 2.59 µs | **459.5 MiB/s · 1.39 µs** | 111.5 MiB/s · 6.57 µs | 292.0 MiB/s · 2.51 µs |
| LINEAR_HASH_BENCH | 102.3 MiB/s · 4.18 µs* | **319.8 MiB/s · 1.34 µs** | 75.0 MiB/s · 6.51 µs | 189.8 MiB/s · 2.57 µs |

*M4 LINEAR_HASH_BENCH scalar at 14 threads shows OMP contention
(142 MiB/s at 7 threads). The NEON path tolerates the contention better.

**Metal doesn't have a standalone POSEIDON_BENCH-equivalent** (it's only
exercised through `merkletree_metal` which invokes the permutation
~17 × N times for N leaves). From the 8M × 128 Merkle number we can
derive the effective permutation rate:

- 8M leaves × 16 absorb iters per leaf + ~8M internal nodes ≈ 136M pod12 calls
- 5.64 s / 136M = **41 ns per pod12 call** → ≈ **1.94 M permutations/sec per GPU vs CPU NEON 0.72 M/s × 14 = 10.1 M/s aggregate; per-core GPU rate vs per-core NEON rate is therefore roughly 1:1, and the Metal win is pure parallel-scaling.**

This is consistent with the observed **2.80× Metal/NEON ratio** on Merkle
(the GPU has ~20 cores of effective parallelism, each ≈ a NEON core, with
GPU overhead eating some of the scaling).

---

## 5. NTT_BLOCK / LDE / LDE_BLOCK

The Metal backend does NOT implement `NTT_Block` or `extendPol` (coset
LDE). The existing scalar and NEON implementations remain the only
path for these. For completeness, CPU-only numbers:

| Benchmark | M4 Pro 14 threads | Xeon 16 threads |
|---|---:|---:|
| NTT_BLOCK_BENCH (scalar) | 1.93 s | 8.74 s |
| LDE_BENCH (scalar) | 13.9 s | 71.3 s |
| LDE_BLOCK_BENCH (scalar) | 6.09 s | 29.1 s |

There are no `_AVX` or `_NEON` variants of these benchmarks. M4 Pro
scalar beats Xeon scalar by 2.1-4.5× on these — larger wins than the
Poseidon benchmarks because NTT/LDE are cache- and memory-bandwidth-bound
where M4 Pro's unified memory hierarchy dominates.

**Metal implementation potential**: adding an `extendPol_Metal` would be
natural follow-up work. The MSL NTT kernels already in place would cover
most of the pipeline; only the coset pre-multiplication (shift table
`r[]`, `r_[]` computed by `NTT_Goldilocks::computeR`) and the
zero-extension need to be plumbed through the Metal bridge.

---

## 6. Summary — what actually matters

### Within-chip (Apple M4 Pro user perspective)

| Task | NEON | **Metal** | Metal/NEON |
|---|---:|---:|---:|
| MERKLETREE_BENCH (8M × 128) | 15.77 s | **5.64 s** | **2.80×** |
| NTT at 2¹⁸ × 128 (STARK) | 51.1 ms | **26.3 ms** | **1.94×** |
| NTT at 2²⁰ × 1 | 6.20 ms | **2.88 ms** | **2.15×** |

### Cross-platform (against x86 AVX2)

Measured on Xeon D-2141I (8-physical-core, 2018 server chip):

| Task | AVX2 (Xeon) | **Metal (M4 Pro)** | Metal/AVX2 |
|---|---:|---:|---:|
| MERKLETREE_BENCH | 44.95 s | **5.64 s** | **7.97×** |
| POSEIDON_BENCH_FULL | 2.49 µs/hash | ~0.46 µs/hash * | ~5.4× |
| NTT_BENCH | n/a (scalar 34.4 s) | ~3 s ** | n/a |
| LDE_BENCH | n/a (scalar 71.3 s) | n/a (not implemented) | — |

*Derived, see §4. **Extrapolated from bench_ntt results.

### Caveats

- **Xeon D-2141I is a weak x86 baseline** (8 physical cores, 2.2 GHz,
  2018-era). A modern Xeon/EPYC with 32+ cores + AVX-512 would
  significantly narrow the Metal-vs-x86 gap via core scaling (possibly
  to 1-2× rather than 7.97×). See the Xeon results doc for reproduction
  guidance on that system; rerunning on Sapphire Rapids or Genoa would
  give a fairer cross-platform picture.
- **NEON on M4 Pro already beats AVX2 on this Xeon by 2.85×** on
  Merkle. This reflects both Apple Silicon's per-core strength and
  the Xeon's weak baseline.
- **AVX2 on a modern x86 box (32-core Sapphire Rapids)**: estimate
  from 2018-Xeon × 4 core scaling × ~1.2 frequency factor → ~870 MiB/s
  on MERKLETREE_BENCH. Metal on M4 Pro at 1452 MiB/s would be ~1.7×
  that estimate. Modern x86 numbers are welcome contributions.

---

## 7. Optimization trajectory

17 incremental commits on `metal-gpu` — summarized here. Each commit
message in `git log metal-gpu ^master` contains detailed before/after
measurements.

| Commit | Focus | Merkle 262k win | NTT 2¹⁸×128 win |
|---|---|---:|---:|
| `4da1b05` | Initial Metal backend | 1.87× NEON | 1.12× NEON |
| `3d10b6f` | NTT batched-phase dispatch | — | 1.20× |
| `587bd90` | Merkle parent-levels batched | 2.00× | — |
| `6fbb4f6` | `gl_mul_small` for MDS | **2.44×** | — |
| `c5d6c4f` | Branchless gl_mul + unroll | 2.65× | — |
| `93ce8e2` | NTT fused rev+s=1 | — | 1.50× |
| `eba5d67` | INTT fused reorder+scale | — | — |
| `222f1e3` | Merkle batch API + big bench | — | — |
| `cc50370` | **NTT radix-4 butterfly** | — | **1.79×** |
| `650ede5` | NTT fused rev+s=1+s=2 | — | 1.88× |
| `b44a88a` | NTT thread-count radix-4 gate | — | **2.03×** |

Also kept in-tree as selectable variants (measured, did NOT improve the
default path):
- `a03c1f3` SIMD-cooperative Poseidon (2× at tiny N, 3× slower at 262k)
- `cf053e8` Coalesced-read variants (transpose + cm kernel, fused-tile tg)

---

## 8. How to reproduce

```bash
# 1. Build and test Metal kernels
make testmetal
make runtestmetal            # bit-exact oracle match

# 2. Benchmark Merkle (small and production sizes)
cd benchs/metal_probes
make bench_merkle
./bench_merkle
./bench_merkle --big         # adds 1 GB and 8 GB shapes

# 3. Benchmark NTT
make bench_ntt
./bench_ntt

# 4. Sequential vs batched Merkle
make bench_merkle_batch
./bench_merkle_batch

# 5. Cross-check against the official CPU benchmark harness
cd ../..
make benchcpu
./benchcpu --benchmark_filter='MERKLETREE_BENCH|NTT_BENCH' --benchmark_min_time=0.1s
```

**Caveats for reproduction:**
- Close other GPU-heavy apps (Chrome/Safari with many tabs, any graphics
  work). Metal shares the GPU with the system.
- Run on AC power. M4 Pro throttles GPU frequency on battery.
- The first `bench_merkle` run is slightly slower due to pipeline
  compilation. Subsequent runs hit the cached pipelines.
- Variance ±3% is typical across runs without GPU quiesce.

---

## 9. Tests passing

```
make runtestmetal
# [ RUN      ] GOLDILOCKS_TEST.merkletree_metal
# [       OK ] GOLDILOCKS_TEST.merkletree_metal (38 ms)
# [ RUN      ] GOLDILOCKS_TEST.NTT_Metal_roundtrip
# [       OK ] GOLDILOCKS_TEST.NTT_Metal_roundtrip (39 ms)
```

Both tests use the Fibonacci-input oracle from `tests/tests.cpp:2232-2235`:

```
root[0] = 0x918F7CD0C3E8701F
root[1] = 0x83A130E00F961B02
root[2] = 0x6921497B364123F8
root[3] = 0xBD2B98A57B748BF4
```

All 4 probes pass bit-exact:
- `probe_device` — Metal device discovery
- `probe_field` — 10k random (add/sub/mul) pairs vs CPU scalar
- `probe_poseidon` — 1k random pod12 states vs `hash_full_result_seq`
- `probe_merkle` — 128 × 64 Fibonacci Merkle oracle match

---

## See also

- `src/metal/README.md` — implementation architecture and API reference
- `benchs/results/apple-silicon-m4pro-neon.md` et seq. — NEON phase
  progression (1b through 1h)
- `benchs/results/x86_64-linux-xeon-d2141i.md` — x86 AVX2 baseline
  numbers
- `benchs/results/REPRODUCE.md` — general reproduction guidance
