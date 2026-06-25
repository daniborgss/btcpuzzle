#ifndef FIELD52_IFMA_H
#define FIELD52_IFMA_H

#include <stdint.h>

// 8-way secp256k1 field ops over the SoA layout Fe8 ([5][8]uint64): 5 limbs,
// each a contiguous block of 8 lanes (one __m512i). Pointers reference 40
// contiguous uint64 (limb k's 8 lanes at base + 8*k). Inputs limbs < 2^52
// (limb 4 < 2^48); outputs are magnitude-1 but may be denormalized.

// field52_mul8: out[l] = a[l] * b[l] mod p, for the 8 lanes.
void field52_mul8(uint64_t *out, const uint64_t *a, const uint64_t *b);

// field52_sqr8: out[l] = a[l]^2 mod p, for the 8 lanes (specialized squaring).
void field52_sqr8(uint64_t *out, const uint64_t *a);

#endif
