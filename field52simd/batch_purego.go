//go:build !avx512ifma

package field52simd

// Pure-Go fallback for the batched field ops: unpack the SoA lanes, run the
// single-lane scalar reference on each, repack. No cgo, builds anywhere. The
// IFMA backend (batch_ifma.go) replaces these with a single vector kernel call.

// MulBatch sets out[l] = a[l]*b[l] mod p for all Lanes lanes.
func MulBatch(out, a, b *Fe8) {
	var av, bv, ov [Lanes]Fe
	UnpackLanes(&av, a)
	UnpackLanes(&bv, b)
	for l := 0; l < Lanes; l++ {
		ov[l].Mul(&av[l], &bv[l])
	}
	PackLanes(out, &ov)
}

// SqrBatch sets out[l] = a[l]^2 mod p for all Lanes lanes.
func SqrBatch(out, a *Fe8) {
	var av, ov [Lanes]Fe
	UnpackLanes(&av, a)
	for l := 0; l < Lanes; l++ {
		ov[l].Sqr(&av[l])
	}
	PackLanes(out, &ov)
}
