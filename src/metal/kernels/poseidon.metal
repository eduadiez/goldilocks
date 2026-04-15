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

// ---- pod12_tg: threadgroup-constants variant of pod12 ----------------------
// Identical algorithm to pod12() above but reads constants from threadgroup
// memory passed in by the kernel. Used by merkle_leaves which pre-loads all
// constants into threadgroup shared cache once per group.
inline void pod12_tg(thread ulong st[12],
                     threadgroup const ulong* tgC,
                     threadgroup const ulong* tgM,
                     threadgroup const ulong* tgP,
                     threadgroup const ulong* tgS) {
    for (int i = 0; i < 12; i++) st[i] = gl_add(st[i], tgC[i]);

    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 12; i++) st[i] = pow7_elem(st[i]);
        int base = (r + 1) * 12;
        for (int i = 0; i < 12; i++) st[i] = gl_add(st[i], tgC[base + i]);
        // mvp_M inline on tgM
        ulong old[12];
        for (int i = 0; i < 12; i++) old[i] = st[i];
        for (int i = 0; i < 12; i++) {
            ulong acc = gl_mul(old[0], tgM[i * 12 + 0]);
            for (int j = 1; j < 12; j++) {
                acc = gl_add(acc, gl_mul(old[j], tgM[i * 12 + j]));
            }
            st[i] = acc;
        }
    }

    for (int i = 0; i < 12; i++) st[i] = pow7_elem(st[i]);
    { int base = 4 * 12;
      for (int i = 0; i < 12; i++) st[i] = gl_add(st[i], tgC[base + i]); }
    {
        ulong old[12];
        for (int i = 0; i < 12; i++) old[i] = st[i];
        for (int i = 0; i < 12; i++) {
            ulong acc = gl_mul(old[0], tgP[i * 12 + 0]);
            for (int j = 1; j < 12; j++) {
                acc = gl_add(acc, gl_mul(old[j], tgP[i * 12 + j]));
            }
            st[i] = acc;
        }
    }

    for (int r = 0; r < 22; r++) {
        st[0] = pow7_elem(st[0]);
        st[0] = gl_add(st[0], tgC[60 + r]);

        int s_base = 23 * r;
        ulong s0 = gl_mul(st[0], tgS[s_base]);
        for (int i = 1; i < 12; i++) {
            s0 = gl_add(s0, gl_mul(st[i], tgS[s_base + i]));
        }
        int s_base2 = 23 * r + 11;
        for (int i = 1; i < 12; i++) {
            ulong wi = gl_mul(st[0], tgS[s_base2 + i]);
            st[i] = gl_add(st[i], wi);
        }
        st[0] = s0;
    }

    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 12; i++) st[i] = pow7_elem(st[i]);
        int base = 82 + r * 12;
        for (int i = 0; i < 12; i++) st[i] = gl_add(st[i], tgC[base + i]);
        ulong old[12];
        for (int i = 0; i < 12; i++) old[i] = st[i];
        for (int i = 0; i < 12; i++) {
            ulong acc = gl_mul(old[0], tgM[i * 12 + 0]);
            for (int j = 1; j < 12; j++) {
                acc = gl_add(acc, gl_mul(old[j], tgM[i * 12 + j]));
            }
            st[i] = acc;
        }
    }

    for (int i = 0; i < 12; i++) st[i] = pow7_elem(st[i]);
    ulong old[12];
    for (int i = 0; i < 12; i++) old[i] = st[i];
    for (int i = 0; i < 12; i++) {
        ulong acc = gl_mul(old[0], tgM[i * 12 + 0]);
        for (int j = 1; j < 12; j++) {
            acc = gl_add(acc, gl_mul(old[j], tgM[i * 12 + j]));
        }
        st[i] = acc;
    }
}

inline void linear_hash_tg(thread ulong out[4],
                           const device ulong* inp,
                           uint size,
                           threadgroup const ulong* tgC,
                           threadgroup const ulong* tgM,
                           threadgroup const ulong* tgP,
                           threadgroup const ulong* tgS) {
    if (size <= 4) {
        for (uint i = 0; i < size; i++) out[i] = inp[i];
        for (uint i = size; i < 4; i++) out[i] = 0UL;
        return;
    }
    ulong state[12];
    uint remaining = size;
    bool first = true;
    while (remaining > 0) {
        if (first) { for (int i = 8; i < 12; i++) state[i] = 0UL; first = false; }
        else       { for (int i = 0; i < 4; i++)  state[8 + i] = out[i]; }
        uint n = (remaining < 8) ? remaining : 8;
        uint offset = size - remaining;
        for (uint i = 0; i < n; i++) state[i] = inp[offset + i];
        for (uint i = n; i < 8; i++) state[i] = 0UL;
        ulong tmp[12];
        for (int i = 0; i < 12; i++) tmp[i] = state[i];
        pod12_tg(tmp, tgC, tgM, tgP, tgS);
        for (int i = 0; i < 12; i++) tmp[i] = gl_canonicalize(tmp[i]);
        for (int i = 0; i < 4; i++) out[i] = tmp[i];
        for (int i = 0; i < 4; i++) state[i] = out[i];
        remaining -= n;
    }
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

// =============================================================================
// SIMD-group-cooperative Poseidon12
// =============================================================================
// Each simdgroup (32 Apple lanes) hashes ONE row. Lanes 0..11 each own one
// state element; lanes 12..31 are "inactive" (they still execute but their
// values are never read by lanes 0..11 because every shuffle/reduce source
// index is < 12).
//
// Why: the one-thread-per-row pod12 keeps a 12-element state[] plus a 12-
// element old[] scratch in mvp_, i.e. ~24 ulong registers per thread. At 64
// threads per threadgroup that crushes register-file occupancy. Cooperative
// version keeps exactly ONE ulong per thread; MDS partners come from
// simd_shuffle.
//
// Layout invariants used below:
//   - `my` is this lane's element, meaningful only when lane < 12.
//   - simd_shuffle(my, k) returns the element held by lane k.
//   - simd_broadcast_first(my) returns lane 0's element.
//   - Control flow must be uniform within the simdgroup, so branches that
//     depend on `lane < 12` use predicated masking, not early-return.
//
// Constants contract: see constants.metal.inc (C[118], M_[144], P_[144],
// S[507]); M_[i*12 + j] == M[j][i] (column-major flat) — same as the
// one-thread-per-row kernel.

// Lane-masked field mul: returns 0 when lane >= 12, else gl_mul(a, b).
// Lets accumulate-via-simd_sum work (addition of 0 is identity); but we
// can't use simd_sum for Goldilocks (it's raw uint64 add, not mod-p), so
// we do manual tree reduce via simd_shuffle (see below). This helper is
// just for "multiply this lane's owned element by the appropriate
// constant", gated.
inline ulong gl_mul_masked(ulong a, ulong b, uint lane, uint limit) {
    return (lane < limit) ? gl_mul(a, b) : 0UL;
}

// MSL `simd_shuffle` / `simd_shuffle_xor` are overloaded for 32-bit and
// smaller types. We pack a ulong into uint2 (lo, hi) for shuffle, then
// repack. This compiles to two native shuffle instructions per call.
inline ulong simd_shuffle_u64(ulong x, ushort lane) {
    uint lo = (uint)(x & 0xFFFFFFFFu);
    uint hi = (uint)(x >> 32);
    lo = simd_shuffle(lo, lane);
    hi = simd_shuffle(hi, lane);
    return ((ulong)hi << 32) | (ulong)lo;
}

inline ulong simd_shuffle_xor_u64(ulong x, ushort mask) {
    uint lo = (uint)(x & 0xFFFFFFFFu);
    uint hi = (uint)(x >> 32);
    lo = simd_shuffle_xor(lo, mask);
    hi = simd_shuffle_xor(hi, mask);
    return ((ulong)hi << 32) | (ulong)lo;
}

// Sum (mod p) across lanes 0..11 using manual tree reduce. Lanes >= 12
// must hold 0 (additive identity in this field). All lanes must execute
// the shuffles; only the final value at lane 0 is authoritative (we
// also end up with the sum broadcast to all lanes via xor-shuffle tree).
inline ulong gl_simd_sum_12(ulong x) {
    x = gl_add(x, simd_shuffle_xor_u64(x, 8));  // lane i += lane i^8
    x = gl_add(x, simd_shuffle_xor_u64(x, 4));
    x = gl_add(x, simd_shuffle_xor_u64(x, 2));
    x = gl_add(x, simd_shuffle_xor_u64(x, 1));
    return x;
}

// Cooperative MDS with M_. Each lane computes its own output element.
//   new_my = sum_j M_[my_idx*12 + j] * old_j  where old_j = shuffle(my, j).
// Lanes 12..31 compute garbage (but harmless — never read).
inline ulong mvp_M_coop(ulong my, uint lane) {
    ulong acc = 0;
    #pragma unroll
    for (uint j = 0; j < 12; j++) {
        ulong old_j = simd_shuffle_u64(my, j);
        acc = gl_add(acc, gl_mul(old_j, M_[lane * 12 + j]));
    }
    return acc;
}

inline ulong mvp_P_coop(ulong my, uint lane) {
    ulong acc = 0;
    #pragma unroll
    for (uint j = 0; j < 12; j++) {
        ulong old_j = simd_shuffle_u64(my, j);
        acc = gl_add(acc, gl_mul(old_j, P_[lane * 12 + j]));
    }
    return acc;
}

// Cooperative pod12. On entry/exit, each lane in [0,12) holds state[lane].
// Returns the lane's new element. Lanes >= 12 are no-ops conceptually.
inline ulong pod12_coop(ulong my, uint lane) {
    // Step 1: add initial round constants C[0..11]
    if (lane < 12) my = gl_add(my, C[lane]);

    // Step 2: 3 full rounds: pow7 + add C[(r+1)*12 + lane] + mvp_M
    #pragma unroll
    for (uint r = 0; r < 3; r++) {
        if (lane < 12) {
            my = pow7_elem(my);
            my = gl_add(my, C[(r + 1) * 12 + lane]);
        }
        my = mvp_M_coop(my, lane);
    }

    // Step 3: transition — pow7 + add + mvp_P (P used ONCE)
    if (lane < 12) {
        my = pow7_elem(my);
        my = gl_add(my, C[4 * 12 + lane]);
    }
    my = mvp_P_coop(my, lane);

    // Step 4: 22 partial rounds — only state[0] raised to 7th.
    // Partial-round structure:
    //   st[0] = pow7(st[0]); st[0] += C[60+r]
    //   s0 = dot(state, S[23r + 0..11])
    //   W_[i] = st[0] * S[23r + 11 + i]  for all i
    //   state[i] += W_[i] for i in 0..11
    //   state[0] = s0   (overwrites the += on lane 0)
    #pragma unroll
    for (uint r = 0; r < 22; r++) {
        // Lane 0: pow7 + add round-constant. Other lanes: leave my alone.
        if (lane == 0) {
            my = pow7_elem(my);
            my = gl_add(my, C[60 + r]);
        }

        // s0 dot product: each lane contributes my * S[23r + lane] for
        // lane < 12; lanes >= 12 contribute 0. Tree reduce.
        ulong dot_contrib = gl_mul_masked(my, S[23 * r + lane], lane, 12);
        ulong s0 = gl_simd_sum_12(dot_contrib);

        // W_[lane] = st[0] * S[23r + 11 + lane]  (lane < 12)
        ulong st0 = simd_shuffle_u64(my, 0);
        if (lane < 12) {
            ulong w_lane = gl_mul(st0, S[23 * r + 11 + lane]);
            my = gl_add(my, w_lane);
        }

        // state[0] = s0 overwrites the += above on lane 0.
        if (lane == 0) my = s0;
    }

    // Step 5: 3 full rounds.  C offset = 82 + r*12 + lane.
    #pragma unroll
    for (uint r = 0; r < 3; r++) {
        if (lane < 12) {
            my = pow7_elem(my);
            my = gl_add(my, C[82 + r * 12 + lane]);
        }
        my = mvp_M_coop(my, lane);
    }

    // Step 6: final pow7 + mvp_M.
    if (lane < 12) my = pow7_elem(my);
    my = mvp_M_coop(my, lane);

    return my;  // caller canonicalizes at kernel exit if observed
}

// ---- kernel: merkle_leaves_simd -------------------------------------------
// One SIMDGROUP per leaf row. Threadgroup size = 32 (one simdgroup).
// Lanes 0..11 absorb one rate/capacity element each; lanes 12..31 are idle
// for most of the absorb loop but still participate in shuffle-based MDS.
//
// Dispatch: num_rows threadgroups × 32 threads. Each threadgroup hashes
// exactly one row.
kernel void merkle_leaves_simd(
    device const ulong* inp   [[ buffer(0) ]],
    device       ulong* tree  [[ buffer(1) ]],
    constant     uint&  ncols [[ buffer(2) ]],
    constant     uint&  dim   [[ buffer(3) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint lane  [[ thread_index_in_simdgroup ]]
) {
    uint size = ncols * dim;
    uint row = tgid;
    device const ulong* row_inp = inp + (ulong)row * (ulong)size;

    // Short-input bypass (size <= CAPACITY=4): just copy + zero-pad.
    if (size <= 4) {
        if (lane < 4) {
            ulong v = (lane < size) ? row_inp[lane] : 0UL;
            tree[row * 4 + lane] = v;
        }
        return;
    }

    // Cooperative sponge absorb. `my` holds this lane's state element.
    ulong my = 0UL;
    uint  remaining = size;
    bool  first = true;

    // cap0..cap3 hold the four "previous capacity" values carried across
    // absorb iterations. Read from lanes 0..3 after each pod12, shuffled
    // into lanes 8..11 at the top of the next iteration.
    while (remaining > 0) {
        uint n = (remaining < 8) ? remaining : 8;
        uint offset = size - remaining;

        // Rate (lanes 0..7): load input or pad with 0.
        ulong rate_val = 0UL;
        if (lane < n) rate_val = row_inp[offset + lane];

        // Capacity (lanes 8..11): first iter zero, else prev state[lane-8].
        ulong cap_val = 0UL;
        if (!first && lane >= 8 && lane < 12) {
            // previous `my` on lanes 0..3 holds the new capacity after pod12
            cap_val = simd_shuffle_u64(my, lane - 8);
        }

        // Compose the new state element for this lane.
        if (lane < 8)         my = rate_val;
        else if (lane < 12)   my = cap_val;
        else                  my = 0UL;  // idle lanes zeroed (for safety)

        my = pod12_coop(my, lane);

        // Canonicalize at sponge boundary (lanes 0..3 are the output).
        if (lane < 12) my = gl_canonicalize(my);

        first = false;
        remaining -= n;
    }

    // Write the final 4-element hash to tree[row*4 .. row*4+3].
    if (lane < 4) {
        tree[row * 4 + lane] = my;
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
