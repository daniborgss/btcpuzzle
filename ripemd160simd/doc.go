// Package ripemd160simd computes RIPEMD160 of fixed 32-byte messages a batch at
// a time, with a CPU-specific backend chosen at build time via build tags:
//
//	-tags avx2    8-way AVX2 (cgo)        — x86-64 with AVX2
//	-tags sse4    4-way SSE2/SSE4 (cgo)   — older x86-64 (added separately)
//	(no tag)      pure-Go fallback        — any platform, no cgo required
//
// Every backend exposes the same API:
//
//	const Lanes int                                        // messages per call
//	func HashBatch(out *[Lanes][20]byte, in *[Lanes][32]byte)
//
// Callers size their buffers with Lanes and let the build tag pick the width
// (8 for AVX2, 4 for SSE4, etc.), so no caller code changes between targets.
package ripemd160simd
