# Research: Fixed-Stage Chunked SFPI Scalar Modulo

## 1. Decision

**[Rejected architecture]** A single always-present fixed-stage chunked reducer
is substantially faster than the 254-stage radix-2 reference, but it still
fails the agreed Blackhole performance gate.

The selected `CHUNK_BITS=16` standalone robust lower bound measured:

| Operation | 0 unsafe lanes | 1 unsafe | 16 unsafe | 32 unsafe |
|---|---:|---:|---:|---:|
| `fmod` | 37979.469 | 37979.469 | 37979.469 | 37979.469 |
| floor remainder | 38683.461 | 38683.461 | 38683.461 | 38683.461 |

This is 13.04x and 11.85x slower than the caching-only kernels. It is a
2.40x/2.38x improvement over the 91k-cycle radix-2 reference, but remains far
above the `>10000 cycles/tile` rejection threshold.

The cached-fast-path plus chunked-fallback wrapper also exceeds the eight local
SFPU registers and is rejected by the compiler. The standalone robust form fits
only after making the reducer in-place and materializing constants lazily. Its
measured cost is therefore a lower bound for the requested combined hybrid.

**Architecture decision:** stop optimizing the masked single-kernel hybrid.
Proceed, if desired, with two separately compiled internal variants:

```text
FastBounded       explicit caller-proven range contract; no classifier/fallback
Robust            correctness-first generic path
```

Do not add an automatic tensor scan at this stage. Production headers remain
unchanged.

## 2. Why a direct chunk multiply is not proof-safe

Let a stage estimate a local integer quotient `q_hat` and form:

```text
r_next = r - q_hat*d.
```

A normal FP32 divisor has a 24-bit significand. A 16- or 20-bit quotient makes
the exact product contain up to 40 or 44 significant bits. The local ISA
documentation states that `SFPMAD` is only partially fused: it retains four
product bits beyond FP32, for 28 effective product bits, rather than an
infinite-precision product.

The inspected ISA sources are:

```text
tt-isa-documentation/Miscellaneous/FMA/README.md
tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md
tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/SFPMAD.md
```

They also establish the normal-domain exclusions used below: denormal inputs
are treated as zero, denormal outputs are flushed, and Wormhole requires an
explicit scheduling gap after a dependent SFPMAD whereas Blackhole normally
stalls automatically.

Consequently, a direct large `q_hat*d` discards exactly the low information the
remainder needs. Zero random failures under an ideal fused model would not make
that implementation correct on Tensix.

## 3. Candidate stage

For positive normal residual `r` and divisor `b`, define:

```text
C = CHUNK_BITS
g = exponent(r) - exponent(b)
s = max(g - (C-1), 0)
d = b * 2^s
x = r * 2^-s
t = x / b = r / d
```

The exponent construction gives:

```text
0 <= t < 2^C.
```

For non-power-of-two `b`, precompute:

```text
rho_up = nextUp(nextUp(RN32(1/b))).
```

Power-of-two divisors keep their exact reciprocal. The local estimate is:

```text
z = RN32(x * rho_up)
q_hat = floor(z).
```

The divisor significand is split into high-to-low components:

```text
b = sum(b_j)
component width W = 28-C.
```

Each component is exponent-scaled by `2^s`, and the candidate performs:

```text
candidate = r
for each component d_j:
    candidate = SFPMAD(-q_hat, d_j, candidate)
if candidate < 0:
    candidate += d
```

Zero low components remain zero; applying `setexp` directly to FP32 zero would
manufacture a power of two and is forbidden.

## 4. One-sided quotient proof

Let `u=2^-24`. For a normal reciprocal, RNE is within one half local ULP of the
exact `1/b`. Moving upward twice gives a conservative relative interval:

```text
1 + 1.5u <= rho_up / (1/b) <= 1 + 5u.
```

Normal RNE multiplication then gives the lower bound:

```text
z/t >= (1+1.5u)(1-u) > 1,
```

so the estimate cannot undershoot the exact local ratio. The upper bound is:

```text
z/t <= (1+5u)(1+u) = 1 + 6u + 5u^2.
```

For every requested `C<=20` and `t<2^C`:

```text
0 <= z-t < 2^20(6u+5u^2) < 0.376 < 1.
```

Therefore:

```text
q_hat in {floor(t), floor(t)+1}.
```

This one-sided result is essential. If `q_hat=floor(t)`, the component
subtractions produce the exact value in `[0,d)`. If it is one too large, they
produce an exact value in `(-d,0]`, and one `+d` correction recovers the exact
remainder. An underestimate would first produce a value in `[d,2d)`, which is
not necessarily representable before correction.

## 5. Partial-SFPMAD product proof

The local quotient has at most `C` significant binary places; the exceptional
estimate `q_hat=2^C` has only one significant bit. Each nonzero divisor
component has at most:

```text
W = 28-C
```

significant places. Thus every exact partial product has at most:

```text
C + W = 28
```

significant bits and fits the documented partially fused SFPMAD product
precision. High-to-low component subtraction cancels at least one component
group at a time. The host model additionally checks that every intermediate
subtraction has at most 24 significant places and is therefore representable as
FP32 before the next component is consumed.

This proof explicitly does not claim correctness when an internal component or
result is subnormal and SFPU FTZ changes it.

## 6. Progress invariant and fixed bound

After the correction, the stage result is the exact mathematical value:

```text
r_next = r mod d
0 <= r_next < d = b*2^s.
```

If the result is nonterminal (`r_next>=b`), its exponent gap satisfies:

```text
g_next <= s = g-C+1,
g-g_next >= C-1.
```

Therefore the exponent gap decreases strictly. Across the largest possible
normal-FP32 exponent gap of 253, the fixed worst-case stage count is:

```text
ceil(253/(C-1)).
```

## 7. Host results

Command:

```bash
python3 tools/sfpi_modulo_chunked_reference.py --random 1000000 \
    --chunk-bits 8 12 16 18 20
```

Coverage per candidate:

- 582 deterministic, exact-multiple, adjacent-ULP, and boundary pairs;
- 357,632 positive finite BF16-dividend pairs over eleven selected divisors;
- 1,000,000 seeded normal-FP32/divisor pairs (`seed=0x5f91`).

The selected divisors include `3`, `5`, `7`, `10`, FP32 `0.1`, FP32 `0.3`,
several powers of two, `2^-64`, and the smallest normal power of two.

| C | Fixed stages | Divisor components | Component subtractions, fixed worst case | Observed stages | Quotient error | Corrections | Minimum progress | Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 37 | 2 x <=20b | 74 | 18 | `[0,+1]` | 1 | 7 | 0 |
| 12 | 23 | 2 x <=16b | 46 | 12 | `[0,+1]` | 1 | 11 | 0 |
| 16 | 17 | 2 x <=12b | 34 | 9 | `[0,+1]` | 1 | 15 | 0 |
| 18 | 15 | 3 x <=10b | 45 | 8 | `[0,+1]` | 1 | 17 | 0 |
| 20 | 14 | 3 x <=8b | 42 | 7 | `[0,+1]` | 1 | 19 | 0 |

For each candidate, 1,349,689 pairs completed inside the proof domain and 8,525
pairs were excluded. Every exclusion was an exact intermediate/result becoming
subnormal and entering the unresolved SFPU FTZ policy. No tested pair hit the
other explicit exclusions (non-finite input, non-normal reciprocal, scaling
overflow, or partial-product overflow).

`CHUNK_BITS=16` was the sole device candidate because it is within the requested
10-20 stage range and minimizes fixed component subtractions and live values.

## 8. Blackhole correctness and raw bits

The final Blackhole matrix passed 36 compile-producer cases. The standalone C16
test-only robust reducer then passed 28 Blackhole runtime cases:

- strict raw-bit diagnostics for `b=3,5,7,10`;
- `fmod` and floor remainder;
- positive and negative dividends;
- positive and negative scalar signs;
- exact multiples and signed zero;
- known FP32 and BF16-exact counterexamples;
- FP32 `0.1` and `0.3`, which exercise a nonzero low divisor component.

The raw diagnostics captured `a`, `b`, RNE reciprocal, upward reciprocal,
`scaled`, `q_hat`, pre/post fast corrections, chunked magnitude, classifier,
and divisor components. Relevant disputed results were:

| `b` | `a` bits | Fast residual on Blackhole | After one `+b` | Correct chunked result |
|---:|---:|---:|---:|---:|
| 3 | `0x4c400002` | `-4` (`0xc0800000`) | `-1` (`0xbf800000`) | `2` (`0x40000000`) |
| 5 | `0x4ca00003` | `-6` (`0xc0c00000`) | `-1` (`0xbf800000`) | `4` (`0x40800000`) |
| 7 | `0x4cdd546e` | `-8` (`0xc1000000`) | `-1` (`0xbf800000`) | `6` (`0x40c00000`) |
| 10 | `0x4d200003` | `-12` (`0xc1400000`) | `-2` (`0xc0000000`) | `8` (`0x41000000`) |

The divisor-5 and divisor-10 fast residuals differ from a separately rounded
multiply-plus-subtract host approximation. These device values reflect the
documented partially fused SFPMAD behavior and supersede that approximation.

## 9. Register pressure and disassembly

The combined cached-fast-path plus C16 fallback failed compilation with an
illegal SFPU spill. The robust candidate compiles only after:

- changing the reducer from return-by-value to an in-place operation;
- materializing reciprocal/component constants lazily;
- limiting live data to the residual, quotient, exponent shift, and one
  component at a time.

No spill appears in the final standalone robust symbols.

| Architecture | `fmod` symbol | Remainder symbol | Dynamic stage body | Replay |
|---|---:|---:|---:|---:|
| Blackhole | 86 encoded instructions | 108 | 66 SFPU + 2 scalar | 0 |
| Wormhole compile-only | 93 | 115 | 73 SFPU + 2 scalar | 0 |

Blackhole uses local registers through `L6`; adding the surrounding hybrid
state still caused the compiler to spill. Wormhole inserts five `sfpnop`
instructions in the dynamic stage and needs two additional instructions because
it lacks Blackhole's negate modifiers. Wormhole evidence is compile-only; no
runtime result is available.

The final Wormhole correctness/performance matrix passed all 36
compile-producer cases. This verifies compilation only, not numerical behavior
or timing on Wormhole hardware.

## 10. Blackhole performance gate

The MATH_ISOLATE probe creates the lane mix from `vConstTileId`, using the same
instruction stream for 0, 1, 16, and 32 unsafe lanes. The equality of all four
measurements directly confirms that lane predication does not skip the fixed
17-stage scalar loop.

| Variant | `fmod` cycles/tile | Floor remainder cycles/tile |
|---|---:|---:|
| Original baseline | 2976.242 | 3328.242 |
| Caching only | 2912.227 | 3264.227 |
| Invalid two-correction upper bound | 1568.227 | 1888.227 |
| Radix-2 reference | 91229.445 | 91901.445 |
| C16 robust lower bound | 37979.469 | 38683.461 |

The C16 candidate removes about 58% of the radix-2 cost but remains 13.04x and
11.85x slower than caching-only. It is rejected under the predefined
`>10000 cycles/tile` rule.

## 11. Separate-variant design

The next architecture should expose an internal compile-time selection:

```cpp
enum class ScalarModuloAlgorithm {
    Robust,
    FastAssumeBounded,
};
```

### `FastAssumeBounded`

Contains only:

```text
scaled = magnitude * RN32(1/b)
q_hat = truncate_positive(scaled)
residual = SFPMAD(-q_hat, b, magnitude)
one correction in each direction
operation-specific sign restoration
```

It has no device classifier and no fallback body. Its caller/planner contract
must already prove every lane is in the normal safe domain, including the
conservative condition corresponding to `scaled==0 || exponent(scaled)<22`, and
must exclude unsupported special/FTZ cases. The known invalid two-correction
measurements, approximately 1568/1888 cycles per tile, are only a performance
upper bound until this contract and dispatch are implemented.

### `Robust`

This is the default for unknown ranges. Initially it can use the correctness-
first exponent-scaled reference for the established normal domain. The C16
algorithm is a shorter research reference, but cannot be called globally
correct until FTZ/subnormal and special-value policy is resolved.

The variants should be distinct compiled kernels selected from existing range
metadata or an explicit internal call-site guarantee. Do not scan tensor values
automatically merely to select the fast kernel.

## 12. Questions for a Tenstorrent SFPI/LLK developer

1. Is there a supported way to reduce a per-lane SFPI predicate to a scalar
   `any-lane` condition and branch on the math RISC-V so the fallback instruction
   stream is completely skipped when all lanes are safe?
2. If not, is the preferred design two compiled variants: a generic robust
   kernel and a fast bounded kernel with an explicit input-range contract?
3. Does TTNN or its planner already carry value-range metadata or an internal
   fast-math specialization that can select `FastAssumeBounded` without a tensor
   scan?
4. Is there a supported portable Wormhole/Blackhole primitive for retaining the
   low product bits of a 16-bit local quotient times a 24-bit FP32 significand,
   or is split-component SFPMAD the intended technique?
5. Is consuming nearly all local registers expected for this pattern, and are
   there recommended SFPI coding forms that prevent the compiler's illegal-spill
   failure when a classifier surrounds an otherwise spill-free reducer?

These questions now have concrete evidence behind them: exact Blackhole raw
bits, a proof-oriented host model, a 91k radix-2 reference, a 38k C16 lower
bound, identical lane-mix timing, and the observed compiler spill.
