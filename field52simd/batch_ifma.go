//go:build avx512ifma

package field52simd

/*
#cgo CFLAGS: -O3 -mavx512f -mavx512vl -mavx512ifma -DBACKEND_AVX512IFMA
#include "field52_ifma.h"
*/
import "C"

import "unsafe"

// MulBatch sets out[l] = a[l]*b[l] mod p for all Lanes lanes, using the 8-way
// AVX-512 IFMA kernel. Fe8 is memory-identical to the C [5][8]uint64 the kernel
// loads as 5 __m512i.
func MulBatch(out, a, b *Fe8) {
	C.field52_mul8(
		(*C.uint64_t)(unsafe.Pointer(&out[0][0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0][0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0][0])),
	)
}

// SqrBatch sets out[l] = a[l]^2 mod p for all Lanes lanes (specialized squaring).
func SqrBatch(out, a *Fe8) {
	C.field52_sqr8(
		(*C.uint64_t)(unsafe.Pointer(&out[0][0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0][0])),
	)
}
