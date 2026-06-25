package field52simd

import "math/big"

// Batched field ops over Fe8 that are cheap enough to stay in pure Go (shared
// by both backends): add, subtract, and modular inverse. Only Mul/Sqr are the
// backend-selected kernel. All keep the invariant that the IFMA Mul/Sqr need:
// every limb < 2^52 (limb 4 < 2^48 + tiny), value < 2^256 + tiny.

// negBias[k] = 4 * (canonical 52-bit limb k of p). Used by SubBatch to compute
// a - b as a + (4p - b) so every limb stays non-negative without a borrow
// (4*p_limb[k] >= any input limb). Value stays a + 4p < 5p < 2^259.
var negBias = func() [5]uint64 {
	t := new(big.Int).Set(P)
	m := new(big.Int).SetUint64(mask52)
	var b [5]uint64
	for k := 0; k < 5; k++ {
		b[k] = new(big.Int).And(t, m).Uint64() * 4
		t.Rsh(t, 52)
	}
	return b
}()

// foldLane folds a single lane's overflow above 2^256 (limb 4 >> 48) back via
// c = 2^32+977, restoring the limb<2^52 / value<2^256+tiny invariant. Used after
// add/sub, where the incoming value is < ~2^259 so one fold suffices.
func foldLane(z *Fe8, l int) {
	mtop := z[4][l] >> 48
	z[4][l] &= (1 << 48) - 1
	v := z[0][l] + mtop*c52 // mtop < 2^4, c52 < 2^33 -> < 2^52
	z[0][l] = v & mask52
	carry := v >> 52
	for k := 1; k < 5 && carry != 0; k++ {
		v = z[k][l] + carry
		z[k][l] = v & mask52
		carry = v >> 52
	}
}

// AddBatch sets out[l] = a[l] + b[l] (mod p, denormalized) for all lanes.
func AddBatch(out, a, b *Fe8) {
	for l := 0; l < Lanes; l++ {
		var carry uint64
		for k := 0; k < 5; k++ {
			v := a[k][l] + b[k][l] + carry
			out[k][l] = v & mask52
			carry = v >> 52
		}
		foldLane(out, l)
	}
}

// SubBatch sets out[l] = a[l] - b[l] (mod p, denormalized) for all lanes.
func SubBatch(out, a, b *Fe8) {
	for l := 0; l < Lanes; l++ {
		var carry uint64
		for k := 0; k < 5; k++ {
			// a + 4p - b: a + negBias >= b so the subtraction never underflows.
			v := a[k][l] + negBias[k] + carry - b[k][l]
			out[k][l] = v & mask52
			carry = v >> 52
		}
		foldLane(out, l)
	}
}

// sqrN sets out = in^(2^n) for n >= 1 (out may alias in).
func sqrN(out, in *Fe8, n int) {
	SqrBatch(out, in)
	for i := 1; i < n; i++ {
		SqrBatch(out, out)
	}
}

// InverseFe8 sets out[l] = a[l]^(-1) mod p for all lanes, via Fermat
// (a^(p-2)) using the standard secp256k1 addition chain. All 8 lanes are
// inverted in parallel through MulBatch/SqrBatch. a[l] == 0 maps to 0.
func InverseFe8(out, a *Fe8) {
	var x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t Fe8

	SqrBatch(&x2, a)
	MulBatch(&x2, &x2, a) // a^(2^2-1)

	SqrBatch(&x3, &x2)
	MulBatch(&x3, &x3, a) // a^(2^3-1)

	sqrN(&x6, &x3, 3)
	MulBatch(&x6, &x6, &x3) // a^(2^6-1)

	sqrN(&x9, &x6, 3)
	MulBatch(&x9, &x9, &x3) // a^(2^9-1)

	sqrN(&x11, &x9, 2)
	MulBatch(&x11, &x11, &x2) // a^(2^11-1)

	sqrN(&x22, &x11, 11)
	MulBatch(&x22, &x22, &x11) // a^(2^22-1)

	sqrN(&x44, &x22, 22)
	MulBatch(&x44, &x44, &x22) // a^(2^44-1)

	sqrN(&x88, &x44, 44)
	MulBatch(&x88, &x88, &x44) // a^(2^88-1)

	sqrN(&x176, &x88, 88)
	MulBatch(&x176, &x176, &x88) // a^(2^176-1)

	sqrN(&x220, &x176, 44)
	MulBatch(&x220, &x220, &x44) // a^(2^220-1)

	sqrN(&x223, &x220, 3)
	MulBatch(&x223, &x223, &x3) // a^(2^223-1)

	// Final window: a^(p-2) = a^(2^256 - 2^32 - 979).
	sqrN(&t, &x223, 23)
	MulBatch(&t, &t, &x22)
	sqrN(&t, &t, 5)
	MulBatch(&t, &t, a)
	sqrN(&t, &t, 3)
	MulBatch(&t, &t, &x2)
	sqrN(&t, &t, 2)
	MulBatch(out, &t, a)
}
