package ripemd160simd

import (
	"bytes"
	"crypto/rand"
	"testing"

	"golang.org/x/crypto/ripemd160"
)

// These tests run against whichever backend the build tags select: `go test`
// exercises the pure-Go fallback, `go test -tags avx2` the AVX2 backend, etc.

func refHash(msg []byte) [20]byte {
	h := ripemd160.New()
	h.Write(msg)
	var out [20]byte
	copy(out[:], h.Sum(nil))
	return out
}

// TestHashBatchMatchesStdlib checks every lane of the active backend against the
// pure-Go reference over many random batches plus the all-zero / all-0xff edges.
func TestHashBatchMatchesStdlib(t *testing.T) {
	var in [Lanes][32]byte
	var out [Lanes][20]byte

	check := func(name string) {
		HashBatch(&out, &in)
		for l := 0; l < Lanes; l++ {
			want := refHash(in[l][:])
			if !bytes.Equal(out[l][:], want[:]) {
				t.Fatalf("%s lane %d:\n  got  %x\n  want %x", name, l, out[l], want)
			}
		}
	}

	check("all-zero")
	for l := 0; l < Lanes; l++ {
		for j := range in[l] {
			in[l][j] = 0xff
		}
	}
	check("all-ff")

	for iter := 0; iter < 200; iter++ {
		for l := 0; l < Lanes; l++ {
			if _, err := rand.Read(in[l][:]); err != nil {
				t.Fatal(err)
			}
		}
		check("random")
	}
}

// BenchmarkBackend measures throughput of the active backend (ns/op is per batch
// of Lanes messages).
func BenchmarkBackend(b *testing.B) {
	var in [Lanes][32]byte
	var out [Lanes][20]byte
	for l := 0; l < Lanes; l++ {
		rand.Read(in[l][:])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		HashBatch(&out, &in)
	}
}

// BenchmarkStdlib is the apples-to-apples baseline: Lanes sequential pure-Go
// RIPEMD160 hashes.
func BenchmarkStdlib(b *testing.B) {
	var in [Lanes][32]byte
	for l := 0; l < Lanes; l++ {
		rand.Read(in[l][:])
	}
	h := ripemd160.New()
	var sum [20]byte
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for l := 0; l < Lanes; l++ {
			h.Reset()
			h.Write(in[l][:])
			h.Sum(sum[:0])
		}
	}
}
