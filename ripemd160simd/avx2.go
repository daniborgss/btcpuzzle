//go:build avx2

package ripemd160simd

/*
#cgo CFLAGS: -O3 -mavx2 -DBACKEND_AVX2
#include "ripemd160_avx2.h"
*/
import "C"

import "unsafe"

// Lanes is how many messages HashBatch processes per call on the AVX2 backend.
const Lanes = 8

// HashBatch computes RIPEMD160 of Lanes 32-byte messages in parallel using the
// 8-way AVX2 implementation. out[l] is the digest of in[l].
func HashBatch(out *[Lanes][20]byte, in *[Lanes][32]byte) {
	C.ripemd160_avx2_8(
		(*C.uint8_t)(unsafe.Pointer(&in[0][0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0][0])),
	)
}
