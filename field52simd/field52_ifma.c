//go:build avx512ifma

// Compiled only for the AVX-512 IFMA backend. Unlike ripemd160simd (whose
// package always has an active cgo file under any cgo build), field52simd has
// no cgo file unless the avx512ifma tag is set, so this .c would be an orphan
// under e.g. -tags avx2 alone. The build constraint above excludes it in that
// case; the #ifdef is a second guard (cgo otherwise compiles every .c).
// BACKEND_AVX512IFMA is defined by batch_ifma.go's cgo CFLAGS.
#ifdef BACKEND_AVX512IFMA

#include "field52_ifma.h"
#include <immintrin.h>

// 8-way secp256k1 field arithmetic in radix 2^52, one element per 64-bit lane.
// This is a literal, op-for-op translation of the scalar reference in
// purego.go (madd52lo/hi, the schoolbook column schedule, and the 3-round
// Solinas reduction) with every op replaced by its __m512i / vpmadd52
// equivalent. The Go reference is the byte-for-byte oracle (see batch_test.go).

#define VLO(acc, a, b) _mm512_madd52lo_epu64((acc), (a), (b)) // acc += low52(a*b)
#define VHI(acc, a, b) _mm512_madd52hi_epu64((acc), (a), (b)) // acc += hi52 (a*b)

// reduce folds the 10 product columns t[0..9] (each lane < 2^52 after the carry
// pass) down to 5 limbs r[0..4] congruent mod p. Mirrors reduceSolinas:
//   c   = 2^32 + 977         (fold at the 2^256 boundary)
//   16c = 68719492368        (fold the high limbs, weight starts at 2^260)
static void reduce(__m512i r[5], const __m512i t[10]) {
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i mask48 = _mm512_set1_epi64((1ULL << 48) - 1);
    const __m512i cAdj = _mm512_set1_epi64(68719492368ULL);
    const __m512i c52 = _mm512_set1_epi64(4294968273ULL);
    __m512i m;

    // ---- Round 1: fold columns 5..9 (weight >= 2^260) with 16c. ----
    m = VLO(t[0], t[5], cAdj);
    r[0] = _mm512_and_si512(m, mask52);

    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), t[1]);
    m = VHI(m, t[5], cAdj);
    m = VLO(m, t[6], cAdj);
    r[1] = _mm512_and_si512(m, mask52);

    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), t[2]);
    m = VHI(m, t[6], cAdj);
    m = VLO(m, t[7], cAdj);
    r[2] = _mm512_and_si512(m, mask52);

    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), t[3]);
    m = VHI(m, t[7], cAdj);
    m = VLO(m, t[8], cAdj);
    r[3] = _mm512_and_si512(m, mask52);

    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), t[4]);
    m = VHI(m, t[8], cAdj);
    m = VLO(m, t[9], cAdj);
    r[4] = _mm512_and_si512(m, mask52);

    __m512i r5 = _mm512_srli_epi64(m, 52); // column 5 (weight 2^260), small
    r5 = VHI(r5, t[9], cAdj);

    // ---- Round 2: fold r5 (weight 2^260) with 16c. ----
    m = VLO(r[0], r5, cAdj);
    r[0] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[1]);
    m = VHI(m, r5, cAdj);
    r[1] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[2]);
    r[2] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[3]);
    r[3] = _mm512_and_si512(m, mask52);
    r[4] = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[4]);

    // ---- Round 3: fold the final overflow above 2^256 with c (unscaled). ----
    __m512i mtop = _mm512_srli_epi64(r[4], 48);
    r[4] = _mm512_and_si512(r[4], mask48);
    m = VLO(r[0], mtop, c52);
    r[0] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[1]);
    m = VHI(m, mtop, c52);
    r[1] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[2]);
    r[2] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[3]);
    r[3] = _mm512_and_si512(m, mask52);
    r[4] = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[4]);
}

// carry_normalize reduces the 10 schoolbook columns to radix 2^52 (each < 2^52).
static void carry_normalize(__m512i t[10]) {
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    __m512i carry = _mm512_setzero_si512();
    for (int k = 0; k < 10; k++) {
        __m512i v = _mm512_add_epi64(t[k], carry);
        t[k] = _mm512_and_si512(v, mask52);
        carry = _mm512_srli_epi64(v, 52);
    }
}

void field52_mul8(uint64_t *out, const uint64_t *a, const uint64_t *b) {
    __m512i av[5], bv[5], t[10];
    for (int k = 0; k < 5; k++) {
        av[k] = _mm512_loadu_si512((const void *)(a + 8 * k));
        bv[k] = _mm512_loadu_si512((const void *)(b + 8 * k));
    }
    for (int k = 0; k < 10; k++) {
        t[k] = _mm512_setzero_si512();
    }
    // Schoolbook 5x5: low52 -> column i+j, hi52 -> column i+j+1.
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            t[i + j] = VLO(t[i + j], av[i], bv[j]);
            t[i + j + 1] = VHI(t[i + j + 1], av[i], bv[j]);
        }
    }
    carry_normalize(t);

    __m512i r[5];
    reduce(r, t);
    for (int k = 0; k < 5; k++) {
        _mm512_storeu_si512((void *)(out + 8 * k), r[k]);
    }
}

void field52_sqr8(uint64_t *out, const uint64_t *a) {
    __m512i av[5], t[10];
    for (int k = 0; k < 5; k++) {
        av[k] = _mm512_loadu_si512((const void *)(a + 8 * k));
    }
    for (int k = 0; k < 10; k++) {
        t[k] = _mm512_setzero_si512();
    }
    // Cross products a[i]*a[j], i<j, accumulated once.
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 5; j++) {
            t[i + j] = VLO(t[i + j], av[i], av[j]);
            t[i + j + 1] = VHI(t[i + j + 1], av[i], av[j]);
        }
    }
    // Double the cross contribution, then add the undoubled diagonals.
    for (int k = 0; k < 10; k++) {
        t[k] = _mm512_slli_epi64(t[k], 1);
    }
    for (int i = 0; i < 5; i++) {
        t[2 * i] = VLO(t[2 * i], av[i], av[i]);
        t[2 * i + 1] = VHI(t[2 * i + 1], av[i], av[i]);
    }
    carry_normalize(t);

    __m512i r[5];
    reduce(r, t);
    for (int k = 0; k < 5; k++) {
        _mm512_storeu_si512((void *)(out + 8 * k), r[k]);
    }
}

#endif // BACKEND_AVX512IFMA
