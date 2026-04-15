#ifndef POSEIDON_GOLDILOCKS_NEON_HPP
#define POSEIDON_GOLDILOCKS_NEON_HPP

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include "simd/goldilocks_simd.hpp"

#ifdef GOLDILOCKS_HAS_NEON
#include <arm_neon.h>

inline void PoseidonGoldilocks::hash_neon(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result_neon(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

inline void PoseidonGoldilocks::pow7_neon(uint64x2_t st[6])
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    // Phase 1j: unrolled in pairs so the compiler / CPU sees two independent
    // mul chains per iteration. Apple Silicon has 2 integer-mul pipes;
    // interleaving exposes them both every cycle instead of stalling on the
    // pw2 -> pw4 dependency.
    for (int i = 0; i < 6; i += 2) {
        auto a0 = st[i];
        auto a1 = st[i + 1];
        auto pw2_0 = N::square_reduced(a0);
        auto pw2_1 = N::square_reduced(a1);
        auto pw4_0 = N::square_reduced(pw2_0);
        auto pw4_1 = N::square_reduced(pw2_1);
        auto pw3_0 = N::mul_reduced(pw2_0, a0);
        auto pw3_1 = N::mul_reduced(pw2_1, a1);
        st[i]     = N::mul_reduced(pw3_0, pw4_0);
        st[i + 1] = N::mul_reduced(pw3_1, pw4_1);
    }
}

inline void PoseidonGoldilocks::add_neon(uint64x2_t st[6], const Goldilocks::Element C[SPONGE_WIDTH])
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    for (int i = 0; i < 6; ++i) {
        auto c = N::load(&C[i * 2]);
        st[i] = N::add(st[i], c);
    }
}

// NEON matrix-vector product: state = M^T * old_state (shape SPONGE_WIDTH x SPONGE_WIDTH).
// Operates entirely in NEON registers; caller provides state both as input (old) and output.
//
// Phase 1l: uses mul_small_reduced for MDS matrix entries (Poseidon M matrix has
// all entries ≤ 41). P matrix (used once per hash) has full 64-bit values and
// still uses mul_reduced.
inline void PoseidonGoldilocks::mvp_neon(Goldilocks::Element *state,
    const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH],
    bool mat_is_small)
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    Goldilocks::Element old_state[SPONGE_WIDTH];
    std::memcpy(old_state, state, SPONGE_WIDTH * sizeof(Goldilocks::Element));

    uint64x2_t out[6];
    for (int i = 0; i < 6; ++i) out[i] = N::splat(0);

    if (mat_is_small) {
        // Fast path: every mat[j][k] fits in 32 bits (Poseidon M).
        for (int j = 0; j < SPONGE_WIDTH; ++j) {
            uint64x2_t bc = N::splat(old_state[j].fe);
            for (int i = 0; i < 6; ++i) {
                uint64_t m0 = mat[j][i * 2].fe;
                uint64_t m1 = mat[j][i * 2 + 1].fe;
                out[i] = N::add(out[i], N::mul_small_reduced(bc, m0, m1));
            }
        }
    } else {
        // General path: mat entries can be full 64-bit (Poseidon P).
        for (int j = 0; j < SPONGE_WIDTH; ++j) {
            uint64x2_t bc = N::splat(old_state[j].fe);
            for (int i = 0; i < 6; ++i) {
                uint64x2_t m_ji = N::load(&mat[j][i * 2]);
                out[i] = N::add(out[i], N::mul_reduced(bc, m_ji));
            }
        }
    }
    for (int i = 0; i < 6; ++i) N::store(&state[i * 2], out[i]);
}

// Phase 1c: 2-hash-parallel NEON matrix-vector product.
// Layout: st[k] = {stA[k], stB[k]} for k=0..11. Processes 2 independent
// Poseidon hashes per NEON register; doubles throughput when merkle rows pair up.
inline void PoseidonGoldilocks::mvp_neon_2(uint64x2_t st[12],
    const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH],
    bool mat_is_small)
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    uint64x2_t old[12];
    for (int k = 0; k < 12; ++k) old[k] = st[k];

    if (mat_is_small) {
        // Fast path: MDS entries fit in 32 bits.
        for (int k = 0; k < 12; ++k) {
            uint64x2_t acc = N::splat(0);
            for (int j = 0; j < 12; ++j) {
                uint64_t m = mat[j][k].fe;
                acc = N::add(acc, N::mul_small_reduced(old[j], m, m));
            }
            st[k] = acc;
        }
    } else {
        // General path (used once per hash for P matrix).
        for (int k = 0; k < 12; ++k) {
            uint64x2_t acc = N::splat(0);
            for (int j = 0; j < 12; ++j) {
                uint64x2_t m_jk = N::splat(mat[j][k].fe);
                acc = N::add(acc, N::mul_reduced(m_jk, old[j]));
            }
            st[k] = acc;
        }
    }
}

// NEON dot product: sum_{i=0..11} x[i] * C[i].
// Lane-parallel multiply-accumulate, horizontal reduce at the end.
inline Goldilocks::Element PoseidonGoldilocks::dot_neon(const Goldilocks::Element *x,
    const Goldilocks::Element C[SPONGE_WIDTH])
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    uint64x2_t acc = N::splat(0);
    for (int i = 0; i < 6; ++i) {
        uint64x2_t xv = N::load(&x[i * 2]);
        uint64x2_t cv = N::load(&C[i * 2]);
        acc = N::add(acc, N::mul_reduced(xv, cv));
    }
    // Canonicalize acc lane values before horizontal reduce
    acc = N::canonicalize(acc);
    Goldilocks::Element lanes[2];
    N::store(lanes, acc);
    Goldilocks::Element r;
    Goldilocks::add(r, lanes[0], lanes[1]);
    return r;
}

#endif // GOLDILOCKS_HAS_NEON
#endif // POSEIDON_GOLDILOCKS_NEON_HPP
