// poseidon.metal — Poseidon12 sponge, Merkle leaf hashing, Merkle parent reduction
//
// Implements:
//   pod12(state[12])          — full Poseidon12 permutation (in-place, lazy-reduce internal)
//   linear_hash(out[4], in*, size) — sponge absorb for arbitrary-length input
//   kernel merkle_leaves       — one thread per leaf row
//   kernel merkle_parents      — one thread per parent node
//
// Constants contract (from Part 3 generated constants.metal.inc):
//   constant ulong C[118]      — round constants (C[0..11] initial, then per-round)
//   constant ulong M_[144]     — MDS matrix, flat row-major M_[row*12+col]
//   constant ulong P_[144]     — partial-round MDS matrix (used ONCE at transition)
//   constant ulong S[507]      — partial-round sparse matrix rows
//
// Poseidon12 parameters (src/poseidon_goldilocks.hpp):
//   SPONGE_WIDTH=12, RATE=8, CAPACITY=4, HASH_SIZE=4
//   HALF_N_FULL_ROUNDS=4, N_PARTIAL_ROUNDS=22
//
// Lazy-reduce contract: gl_mul returns [0,2p); canonicalize only at kernel exit.

#include <metal_stdlib>
#include "field.metal"
#include "constants.metal.inc"
using namespace metal;

// ---- Poseidon12 helper device functions -----------------------------------

// pow7: x^7 = x * x^2 * x^4 (3 muls, lazy-reduce — no canonicalize)
inline ulong pow7_elem(ulong x) {
    ulong x2 = gl_mul(x,  x);
    ulong x3 = gl_mul(x,  x2);
    ulong x4 = gl_mul(x2, x2);
    return gl_mul(x3, x4);
}

// mvp_M: state[i] = sum_j M[j][i] * old_state[j]
// CPU `mvp_` at poseidon_goldilocks.hpp:193-211 accesses `mat[j][i]`.
// Flat M_[144] is stored **column-major**: M_[col*12 + row] == M[row][col].
// (Verified empirically: M_[0..11] = {0x19,0x0f,0x29,0x10,...} = column 0 of M.)
// So M[j][i] maps to M_[i*12 + j].
// No canonicalize — outputs are lazy-reduced sums of lazy-reduce mul products.
inline void mvp_M(thread ulong st[12]) {
    ulong old[12];
    for (int i = 0; i < 12; i++) old[i] = st[i];
    for (int i = 0; i < 12; i++) {
        ulong acc = gl_mul(old[0], M_[i * 12 + 0]);
        for (int j = 1; j < 12; j++) {
            acc = gl_add(acc, gl_mul(old[j], M_[i * 12 + j]));
        }
        st[i] = acc;
    }
}

// mvp_P: identical layout but uses P_ matrix (also column-major in flat form).
// Called exactly ONCE at the transition from initial full rounds to partial rounds.
inline void mvp_P(thread ulong st[12]) {
    ulong old[12];
    for (int i = 0; i < 12; i++) old[i] = st[i];
    for (int i = 0; i < 12; i++) {
        ulong acc = gl_mul(old[0], P_[i * 12 + 0]);
        for (int j = 1; j < 12; j++) {
            acc = gl_add(acc, gl_mul(old[j], P_[i * 12 + j]));
        }
        st[i] = acc;
    }
}

// pod12: full Poseidon12 permutation (in-place).
// Mirrors hash_full_result_seq (poseidon_goldilocks.cpp:5-37) exactly.
// Canonicalize is done by the caller at kernel exit.
inline void pod12(thread ulong st[12]) {
    // Step 1: add initial round constants C[0..11]
    for (int i = 0; i < 12; i++) {
        st[i] = gl_add(st[i], C[i]);
    }

    // Step 2: first HALF_N_FULL_ROUNDS-1 = 3 full rounds
    // Each: pow7 all 12, add C[(r+1)*12..], mvp_M
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 12; i++) st[i] = pow7_elem(st[i]);
        int base = (r + 1) * 12;
        for (int i = 0; i < 12; i++) st[i] = gl_add(st[i], C[base + i]);
        mvp_M(st);
    }

    // Step 3: transition round — pow7 + add + mvp_P (P used ONCE)
    for (int i = 0; i < 12; i++) st[i] = pow7_elem(st[i]);
    {
        int base = 4 * 12;  // HALF_N_FULL_ROUNDS * SPONGE_WIDTH = 4*12 = 48
        for (int i = 0; i < 12; i++) st[i] = gl_add(st[i], C[base + i]);
    }
    mvp_P(st);

    // Step 4: N_PARTIAL_ROUNDS = 22 partial rounds (only state[0] raised to 7)
    // S array stride: 2*SPONGE_WIDTH - 1 = 23 per round
    // - dot uses S[23*r .. 23*r+11]  (first 12 elements of row)
    // - prod uses S[23*r+11 .. 23*r+22] (last 12 elements of row, 11 shared w/ dot end)
    // C offset for partial rounds: (HALF_N_FULL_ROUNDS+1)*12 + r = 5*12 + r = 60+r
    for (int r = 0; r < 22; r++) {
        st[0] = pow7_elem(st[0]);
        st[0] = gl_add(st[0], C[60 + r]);  // C[(HALF+1)*12 + r] = C[5*12+r]

        // dot_(state, S[23*r..]) over 12 elements
        int s_base = 23 * r;
        ulong s0 = gl_mul(st[0], S[s_base]);
        for (int i = 1; i < 12; i++) {
            s0 = gl_add(s0, gl_mul(st[i], S[s_base + i]));
        }

        // prod_(W_, state[0], S[23*r+11..]) — produces W_[0..11]
        // W_[i] = state[0] * S[23*r + 11 + i]
        // then state[i] += W_[i] for i=1..11; state[0] = s0
        int s_base2 = 23 * r + 11;  // SPONGE_WIDTH - 1 = 11
        for (int i = 1; i < 12; i++) {
            ulong wi = gl_mul(st[0], S[s_base2 + i]);
            st[i] = gl_add(st[i], wi);
        }
        // W_[0] = state[0]*S[s_base2] but state[0] is replaced by s0, not added
        // (prod_ fills W_[0..11], add_ adds to state[0..11], then state[0]=s0 overwrites)
        // So state[0] effectively becomes s0:
        st[0] = s0;
    }

    // Step 5: second HALF_N_FULL_ROUNDS-1 = 3 full rounds
    // C offset: (HALF+1)*12 + N_PARTIAL + r*12 = 5*12+22 + r*12 = 82+r*12
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 12; i++) st[i] = pow7_elem(st[i]);
        int base = 82 + r * 12;  // (HALF_N_FULL_ROUNDS+1)*12 + N_PARTIAL_ROUNDS + r*12
        for (int i = 0; i < 12; i++) st[i] = gl_add(st[i], C[base + i]);
        mvp_M(st);
    }

    // Step 6: final pow7 all 12 + mvp_M (no add_ before final mvp)
    for (int i = 0; i < 12; i++) st[i] = pow7_elem(st[i]);
    mvp_M(st);

    // NO canonicalize here — caller does it at kernel exit
}

// ---- linear_hash: sponge absorb over device buffer -------------------------
// Mirrors linear_hash_seq (poseidon_goldilocks.cpp:38-73).
// size == number of ulong elements.
// output must have space for 4 elements (CAPACITY = HASH_SIZE = 4).
inline void linear_hash(thread ulong out[4],
                        const device ulong* inp,
                        uint size) {
    if (size <= 4) {  // size <= CAPACITY
        for (uint i = 0; i < size; i++) out[i] = inp[i];
        for (uint i = size; i < 4; i++) out[i] = 0UL;
        return;
    }

    ulong state[12];
    uint remaining = size;
    bool first = true;

    while (remaining > 0) {
        // Set capacity portion of state
        if (first) {
            for (int i = 8; i < 12; i++) state[i] = 0UL;
            first = false;
        } else {
            // copy previous capacity (state[0..3]) into state[8..11]
            for (int i = 0; i < 4; i++) state[8 + i] = out[i];
        }

        uint n = (remaining < 8) ? remaining : 8;
        uint offset = size - remaining;
        for (uint i = 0; i < n; i++) state[i] = inp[offset + i];
        for (uint i = n; i < 8; i++) state[i] = 0UL;

        // Run permutation
        ulong tmp[12];
        for (int i = 0; i < 12; i++) tmp[i] = state[i];
        pod12(tmp);
        // Canonicalize at sponge boundary
        for (int i = 0; i < 12; i++) tmp[i] = gl_canonicalize(tmp[i]);
        // Save output (capacity = first 4 for next round)
        for (int i = 0; i < 4; i++) out[i] = tmp[i];
        // Update state capacity for next iteration
        for (int i = 0; i < 4; i++) state[i] = out[i];

        remaining -= n;
    }
}

// ---- kernel: merkle_leaves -------------------------------------------------
// One thread per leaf row i.
// Input:  inp[num_rows * ncols * dim]  (row-major, row i at inp[i*ncols*dim])
// Output: tree[num_rows * 4]           (row i at tree[i*4])
kernel void merkle_leaves(
    device const ulong* inp   [[ buffer(0) ]],
    device       ulong* tree  [[ buffer(1) ]],
    constant     uint&  ncols [[ buffer(2) ]],
    constant     uint&  dim   [[ buffer(3) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint size = ncols * dim;
    ulong out[4];
    linear_hash(out, inp + tid * size, size);
    for (int i = 0; i < 4; i++) {
        tree[tid * 4 + i] = out[i];
    }
}

// ---- kernel: merkle_parents ------------------------------------------------
// One thread per parent node i (0-based).
// tree[nextIndex .. nextIndex + pending*4) = children at this level.
// parent[i] = hash_seq(children[2i] ++ children[2i+1], zeros[8..11])
// For odd pending: last parent copies last child (no new hash needed).
//
// Caller passes:
//   buf        = full tree buffer
//   nextIndex  = flat element offset to start of this level's children
//   pending    = number of children nodes at this level
kernel void merkle_parents(
    device ulong*    buf        [[ buffer(0) ]],
    constant uint&   nextIndex  [[ buffer(1) ]],
    constant uint&   pending    [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    // Children start at buf[nextIndex]; parent output at buf[nextIndex + pending*4]
    uint child_off = nextIndex + tid * 8;    // each child is 4 ulongs; pair = 8
    uint parent_off = nextIndex + pending * 4 + tid * 4;

    if (pending % 2 == 1 && tid == (pending / 2)) {
        // Odd last node: copy last child to parent
        uint last_child = nextIndex + (pending - 1) * 4;
        for (int i = 0; i < 4; i++) buf[parent_off + i] = buf[last_child + i];
        return;
    }

    // Build pol_input[0..7] = left_child[0..3] ++ right_child[0..3]
    // pol_input[8..11] = 0
    ulong pol[12];
    for (int i = 0; i < 8; i++) pol[i] = buf[child_off + i];
    for (int i = 8; i < 12; i++) pol[i] = 0UL;

    // Run pod12 (in-place on pol)
    pod12(pol);
    // Canonicalize at kernel exit
    for (int i = 0; i < 4; i++) buf[parent_off + i] = gl_canonicalize(pol[i]);
}
