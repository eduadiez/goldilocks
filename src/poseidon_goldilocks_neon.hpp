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
    for (int i = 0; i < 6; ++i) {
        auto pw2 = N::square(st[i]);
        auto pw4 = N::square(pw2);
        auto pw3 = N::mul(pw2, st[i]);
        st[i] = N::mul(pw3, pw4);
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
inline void PoseidonGoldilocks::mvp_neon(Goldilocks::Element *state,
    const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH])
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    Goldilocks::Element old_state[SPONGE_WIDTH];
    std::memcpy(old_state, state, SPONGE_WIDTH * sizeof(Goldilocks::Element));

    uint64x2_t out[6];
    for (int i = 0; i < 6; ++i) out[i] = N::splat(0);

    for (int j = 0; j < SPONGE_WIDTH; ++j) {
        uint64x2_t bc = N::splat(old_state[j].fe);
        for (int i = 0; i < 6; ++i) {
            uint64x2_t m_ji = N::load(&mat[j][i * 2]);
            out[i] = N::add(out[i], N::mul(bc, m_ji));
        }
    }
    for (int i = 0; i < 6; ++i) N::store(&state[i * 2], out[i]);
}

// Phase 1c: 2-hash-parallel NEON matrix-vector product.
// Layout: st[k] = {stA[k], stB[k]} for k=0..11. Processes 2 independent
// Poseidon hashes per NEON register; doubles throughput when merkle rows pair up.
inline void PoseidonGoldilocks::mvp_neon_2(uint64x2_t st[12],
    const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH])
{
    using N = goldilocks::simd::GLSimd<goldilocks::simd::Neon>;
    uint64x2_t old[12];
    for (int k = 0; k < 12; ++k) old[k] = st[k];

    for (int k = 0; k < 12; ++k) {
        uint64x2_t acc = N::splat(0);
        for (int j = 0; j < 12; ++j) {
            uint64x2_t m_jk = N::splat(mat[j][k].fe);
            acc = N::add(acc, N::mul(m_jk, old[j]));
        }
        st[k] = acc;
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
        acc = N::add(acc, N::mul(xv, cv));
    }
    Goldilocks::Element lanes[2];
    N::store(lanes, acc);
    Goldilocks::Element r;
    Goldilocks::add(r, lanes[0], lanes[1]);
    return r;
}

#endif // GOLDILOCKS_HAS_NEON
#endif // POSEIDON_GOLDILOCKS_NEON_HPP
