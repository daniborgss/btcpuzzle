package field52simd

import (
	"math/big"
	"testing"
)

// These validate the fused EC steps (one cgo call over all groups) against the
// same computation done with the individually-tested batch ops. Under -tags
// avx512ifma this checks the C fusion; under no tag it checks the Go fallback.

func randGroups(t *testing.T, ng int) []Fe8 {
	t.Helper()
	g := make([]Fe8, ng)
	for i := range g {
		s, _ := randFe8(t)
		g[i] = s
	}
	return g
}

func unpackVal(g *Fe8, lane int) *big.Int {
	var fe [Lanes]Fe
	UnpackLanes(&fe, g)
	b := fe[lane].Bytes()
	return new(big.Int).SetBytes(b[:])
}

func TestSlopeSetupFused(t *testing.T) {
	const ng = 3
	x, y := randGroups(t, ng), randGroups(t, ng)
	xG, _ := randFe8(t)
	yG, _ := randFe8(t)
	denom := make([]Fe8, ng)
	num := make([]Fe8, ng)
	SlopeSetup(denom, num, x, y, &xG, &yG)
	for g := 0; g < ng; g++ {
		var wantD, wantN Fe8
		SubBatch(&wantD, &x[g], &xG)
		SubBatch(&wantN, &y[g], &yG)
		for l := 0; l < Lanes; l++ {
			if unpackVal(&denom[g], l).Cmp(unpackVal(&wantD, l)) != 0 ||
				unpackVal(&num[g], l).Cmp(unpackVal(&wantN, l)) != 0 {
				t.Fatalf("SlopeSetup group %d lane %d mismatch", g, l)
			}
		}
	}
}

// TestMontInversionFused checks that forward+inverse+backward yields a true
// inverse: denom[g] * inv[g] == 1 for every group and lane.
func TestMontInversionFused(t *testing.T) {
	const ng = 4
	denom := randGroups(t, ng) // random elements are ~never 0
	prefix := make([]Fe8, ng)
	inv := make([]Fe8, ng)
	var acc, invAcc Fe8
	MontForward(prefix, &acc, denom)
	InverseFe8(&invAcc, &acc)
	MontBackward(inv, &invAcc, prefix, denom)

	one := bytesOf(big.NewInt(1))
	for g := 0; g < ng; g++ {
		var prod Fe8
		MulBatch(&prod, &denom[g], &inv[g])
		var fe [Lanes]Fe
		UnpackLanes(&fe, &prod)
		for l := 0; l < Lanes; l++ {
			if fe[l].Bytes() != one {
				t.Fatalf("MontInversion group %d lane %d: denom*inv != 1", g, l)
			}
		}
	}
}

func TestPointAddFused(t *testing.T) {
	const ng = 3
	x, y := randGroups(t, ng), randGroups(t, ng)
	num, inv, xsub := randGroups(t, ng), randGroups(t, ng), randGroups(t, ng)

	// Reference: the same formula with individual batch ops.
	wantX := append([]Fe8(nil), x...)
	wantY := append([]Fe8(nil), y...)
	for g := 0; g < ng; g++ {
		var lambda, sq, x3, tt, y3 Fe8
		MulBatch(&lambda, &num[g], &inv[g])
		SqrBatch(&sq, &lambda)
		SubBatch(&x3, &sq, &wantX[g])
		SubBatch(&x3, &x3, &xsub[g])
		SubBatch(&tt, &wantX[g], &x3)
		MulBatch(&y3, &lambda, &tt)
		SubBatch(&y3, &y3, &wantY[g])
		wantX[g] = x3
		wantY[g] = y3
	}

	PointAdd(x, y, num, inv, xsub)

	for g := 0; g < ng; g++ {
		for l := 0; l < Lanes; l++ {
			if unpackVal(&x[g], l).Cmp(unpackVal(&wantX[g], l)) != 0 ||
				unpackVal(&y[g], l).Cmp(unpackVal(&wantY[g], l)) != 0 {
				t.Fatalf("PointAdd group %d lane %d mismatch", g, l)
			}
		}
	}
}
