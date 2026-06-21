//go:build !avx2 && !sse4

package ripemd160simd

import "golang.org/x/crypto/ripemd160"

// Lanes for the pure-Go fallback. There is no SIMD here — the messages in a
// batch are simply hashed one after another — so the value only sets the batch
// size the caller works in; 8 matches the AVX2 backend.
const Lanes = 8

// HashBatch computes RIPEMD160 of Lanes 32-byte messages with the pure-Go
// hasher. No cgo and no SIMD, so it builds and runs on any platform; it is the
// correctness reference and the fallback when no backend tag is given.
func HashBatch(out *[Lanes][20]byte, in *[Lanes][32]byte) {
	h := ripemd160.New()
	for l := 0; l < Lanes; l++ {
		h.Reset()
		h.Write(in[l][:])
		h.Sum(out[l][:0])
	}
}
