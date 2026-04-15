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

#endif // GOLDILOCKS_HAS_NEON
#endif // POSEIDON_GOLDILOCKS_NEON_HPP
