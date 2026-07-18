# Correctness Audit: Scalar SFPI `fmod` and `remainder`

## 1. Executive conclusion

**[Proven]** The claim that one positive and one negative residual correction
are sufficient for every supported FP32/BF16 input is false.

The two-correction map is correct for every exact residual `0 <= r < b` only
when the estimated quotient error is

```text
k = q_hat - q in {-1, 0, 1}.
```

**[Empirically verified on host]** Valid FP32 and BF16-input cases exist with
`k = 2`. A compact FP32 counterexample is:

```text
a          = 50331656.0       bits 0x4c400002
b          = 3.0              bits 0x40400000
rho        = fl(1 / b)        bits 0x3eaaaaab
scaled     = fl(a * rho)      16777220.0, bits 0x4b800002
q          = floor(a / b)     16777218
q_hat      = trunc(scaled)    16777220
k          = q_hat - q        2
residual   = -4.0             bits 0xc0800000
corrected  = -1.0             bits 0xbf800000
exact r    = 2.0
```

The corrected residual is still negative and therefore violates
`0 <= r < b`.

**Recommendation: C.** A different range-reduction algorithm is required for a
globally correct implementation. Until that algorithm is implemented, only the
common-subexpression optimization that caches `scaled` is justified as a
semantics-preserving production change. The proposed two-correction replacement
must not be treated as globally correct.

## 2. Classification vocabulary

Important conclusions use these labels:

- **[Proven]** follows mathematically from stated assumptions;
- **[Empirically verified on host]** reproduced with exact FP32 bit patterns and
  exact rational reference arithmetic;
- **[Empirically verified on Blackhole]** observed on Blackhole hardware;
- **[Empirically verified on Wormhole]** observed on Wormhole hardware;
- **[Inferred]** supported by source, ISA, or disassembly but not directly
  measured for the adversarial case;
- **[Unknown]** evidence is not yet available.

## 3. Exact API semantics

**[Proven from repository code and golden tests]** Scalar `ttnn.fmod(x, y)` uses
truncation semantics:

```text
fmod(x, y) = x - trunc(x / y) * y.
```

The nonzero result has the dividend's sign.

**[Proven from repository code and golden tests]** Scalar
`ttnn.remainder(x, y)` uses Python/PyTorch floor-remainder semantics:

```text
remainder(x, y) = x - floor(x / y) * y.
```

The nonzero result has the divisor's sign. This is not C++
`std::remainder`, which rounds the quotient to the nearest integer.

For both operations, the magnitude reduction can be analyzed using positive
`a = abs(x)` and `b = abs(y)`. Operation-specific sign restoration occurs after
the magnitude residual is constructed.

## 4. Current quotient model

For finite `a >= 0` and `b > 0`, define the exact values

```text
t = a / b
q = floor(t)
r = a - q*b,  with 0 <= r < b.
```

The scalar is passed to the generated kernel together with a host-computed FP32
reciprocal:

```text
rho     = fl32(1 / b)
scaled  = fl32(a * rho)
q_hat   = trunc_positive(scaled)
```

The generated SFPU residual uses an `SFPMAD`-class operation corresponding to
`a - q_hat*b`.

## 5. Formal derivation in exact residual arithmetic

Let

```text
q_hat = q + k
```

for an arbitrary integer `k`. Before floating-point rounding of the residual,

```text
r_hat = a - q_hat*b
      = a - (q+k)*b
      = r - k*b.
```

Define the disputed sequential correction map:

```text
C(z):
    if z >= b: z = z - b
    if z <  0: z = z + b
```

For every `r in [0,b)`:

| `k` | Initial `r_hat` | Result of `C` | Correct for all `r`? |
|---:|---|---|---|
| `-1` | `r+b` | first correction gives `r` | yes |
| `0` | `r` | unchanged | yes |
| `1` | `r-b` | second correction gives `r` | yes |
| `>=2` | `r-kb` | at most one `+b`; remains negative | no |
| `<=-2` | `r-kb` | at most one `-b`; remains `>=b` | no |

**[Proven]** One correction in each direction is sufficient for all exact
residuals if and only if `k` is in `{-1,0,1}`.

For a particular residual, floating-point coincidences can mask a larger error,
but they cannot provide a general guarantee.

## 6. Floating-point error in `scaled`

Let binary32 unit roundoff be

```text
u = 2^-24.
```

For normal finite operations under round-to-nearest-even, model

```text
rho    = (1/b) * (1 + delta_1), |delta_1| <= u
scaled = (a*rho) * (1 + delta_2), |delta_2| <= u.
```

Then

```text
scaled = t * (1 + delta_1) * (1 + delta_2)
```

and

```text
|scaled - t| <= t * (2u + u^2).
```

This is a relative error bound. The corresponding absolute error grows with the
quotient magnitude `t`.

At approximately `t = 2^23`, the upper bound approaches one quotient unit. At
approximately `t = 2^24`, it approaches two quotient units. Therefore a uniform
one-unit quotient bound cannot hold for arbitrarily large exponent gaps.

**[Proven under normal IEEE assumptions]** A sufficient condition for
`|scaled-t| < 1` is

```text
t < 1 / (2u + u^2),
```

which is slightly below `2^23`.

This condition is not directly a complete Tenstorrent correctness condition:

- it excludes underflow, overflow, and FTZ behavior;
- it bounds `scaled`, not the partially fused residual;
- exact API output can still differ through residual rounding;
- checking exact `t` would itself require division.

**[Inferred]** A conservative exponent-gap fast path can be built, but the exact
threshold must account for significands, reciprocal normality, and SFPU FTZ.
Using only `scaled_exp < 22` is substantially easier to justify than using
`scaled_exp < 23`, but it is conservative and has not yet been proven bit-exact
against all SFPU edge behavior.

## 7. Host counterexamples

All values in this table are exact FP32 bit patterns. The true quotient and
remainder were computed from exact rational representations of those patterns.

| `b` | `a` | `q` | `q_hat` | `k` | residual before | after two corrections | exact `r` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `3` | `50331656` | `16777218` | `16777220` | `2` | `-4` | `-1` | `2` |
| `5` | `83886104` | `16777220` | `16777222` | `2` | `-8` | `-3` | `4` |
| `7` | `116040560` | `16577222` | `16577224` | `2` | `-8` | `-1` | `6` |
| `10` | `167772208` | `16777220` | `16777222` | `2` | `-16` | `-6` | `8` |
| FP32 `0.1` | `1677722.125` | `16777220` | `16777222` | `2` | `-0.125` | about `-0.025` | FP32 `0.1` |
| FP32 `0.3` | `5033166.5` | `16777220` | `16777222` | `2` | `-0.5` | about `-0.2` | about `0.3` |

**[Empirically verified on host]** The failure is deterministic and begins in
ordinary finite normal ranges around quotient magnitude `2^24`; it does not
require NaN, infinity, subnormal input, or overflow.

### BF16 input counterexample

The following dividend is exactly representable in BF16 and then widened to
FP32:

```text
a BF16 bits = 0x4c42
a FP32      = 50855936.0, bits 0x4c420000
b           = 3.0
scaled      = 16951980.0
q           = 16951978
q_hat       = 16951980
k           = 2
```

**[Empirically verified on host]** Reduced BF16 input precision does not restore
the global one-correction guarantee.

The searches were adversarial and randomized, not exhaustive over all `2^64`
FP32 pairs. No exhaustive claim is made.

## 8. SFPU behavior and on-device status

From the local ISA documentation:

- **[Proven from ISA contract]** denormal `SFPMAD` inputs are treated as zero;
- **[Proven from ISA contract]** denormal results are flushed to zero;
- **[Proven from ISA contract]** `SFPMAD` is partially, not completely, fused;
- **[Proven from ISA contract]** it performs one RNE rounding step;
- **[Proven from ISA contract]** Blackhole preserves the sign when flushing a
  denormal output, while Wormhole documents different negative-zero behavior;
- **[Proven from ISA contract]** multiply-by-one or add-zero matches standalone
  multiply/add aside from documented denormal behavior.

**[Inferred]** The quotient counterexamples should survive on Blackhole because
they use finite normal operands, the reciprocal is programmed by the host, and
the quotient error exists before residual formation.

**[Unknown]** Raw intermediate bit patterns for these adversarial cases have not
yet been returned by a dedicated Blackhole diagnostic kernel.

**[Unknown]** Wormhole hardware results are unavailable on the current server.

Existing Blackhole tests with divisor `2.0` passed, but they do not exercise the
inexact-reciprocal counterexamples and therefore are not device confirmation of
the disputed global claim.

## 9. Analysis of the old ten-correction implementation

The baseline correction is approximately:

```text
repeat 10 times:
    if residual >= b:
        residual = b - residual
```

This is not repeated modulo subtraction. If `residual` is in `[b,2b)`, the map
produces a value in `(-b,0]`, not the expected `residual-b` in `[0,b)`.

If the quotient is too large, the residual is negative and the condition never
executes. The `b=3`, `a=50331656` counterexample enters with `-4`, so all ten
blocks are no-ops.

**[Proven]** The old loop does not converge generally to the correct modulo
interval.

**[Proven]** The number ten is not supported by a quotient-error bound in the
implementation.

**[Empirically verified on host]** The old implementation also fails for large
finite ratios and inexact reciprocals.

The baseline should therefore be understood as existing behavior, not a correct
fallback algorithm.

## 10. Test coverage audit

The focused LLK unary harness programs:

```text
b = 2.0
rho = 0.5
```

Both are exact powers of two. This hides reciprocal-rounding error.

TTNN tests include several decimal scalar values, but primarily BF16 inputs,
moderate magnitudes, and tolerant comparisons. They document inconsistencies at
exact multiples but do not target quotient magnitudes near and above `2^23`.

Required additions:

| Dimension | Values |
|---|---|
| Divisors | powers of two, `3`, `5`, `7`, `10`, FP32 `0.1`, `0.3`, `0.003` |
| Ratio | below `2^20`, around `2^23`, `2^24`, `2^31`, large exponent gaps |
| Position | exact multiple, one ULP below, one ULP above |
| Signs | both dividend and divisor signs |
| Special | signed zero, NaN, infinity, smallest normal, subnormal/FTZ |
| Formats | FP32 input/dest and BF16 input widened to FP32 |

## 11. Recommended algorithm

### Immediate production-safe action

**[Proven]** Caching

```cpp
const vFloat scaled = v * recip_val;
```

is safe with respect to repeated multiplication because identical SFPU inputs
produce the same deterministic multiplication result. It removes redundant
`sfpmul` instructions without changing the quotient value.

Retain baseline correction behavior until a robust reducer is implemented. This
does not make the baseline mathematically correct, but it avoids shipping the
new, unproven global claim as a replacement.

### Globally correct direction

Recommendation C is an exponent-scaled reduction:

1. Compare exponents of `a` and `b`.
2. Scale `b` by the largest power of two not exceeding `a`.
3. Conditionally subtract scaled divisors from largest to smallest.
4. Produce a residual in `[0,b)` without relying on a rounded reciprocal
   quotient.
5. Apply `fmod` or floor-remainder sign semantics afterward.

This is analogous to binary long division and naturally handles large exponent
gaps. It should be specialized so common bounded ratios use a short fast path.

### Possible bounded fast path

A future fast path may retain reciprocal multiplication when all of these hold:

- `a`, `b`, `rho`, and `scaled` are finite normal values;
- no FTZ, overflow, or reciprocal overflow occurred;
- a conservative exponent/magnitude condition proves `|q_hat-q| <= 1`;
- residual rounding is accounted for in the final semantic comparison.

**[Unknown]** The optimal implementable exponent threshold and fallback
instruction schedule require a bit-accurate SFPU model plus device validation.
It must not be guessed from the approximate `2^23` observation alone.

Reciprocal refinement alone is insufficient as a global solution: even a
correctly rounded FP32 reciprocal has relative error whose absolute quotient
effect grows with `a/b`.

## 12. Performance impact

The rejected two-correction candidate measured on Blackhole:

| Operation | Baseline | Candidate | Candidate speedup |
|---|---:|---:|---:|
| `fmod` | 2976.242 cycles/tile | 1568.227 | 1.898x |
| `remainder` | 3328.242 cycles/tile | 1888.227 | 1.763x |

**[Empirically verified on Blackhole]** These are real performance measurements
for the tested divisor `2.0`.

They do not establish correctness for other divisors. The performance result
must be labeled as a rejected or conditional candidate, not as an accepted
global optimization.

Disassembly showed baseline versus candidate:

```text
encoded instructions: 66 -> 59
sfpmul:               4 -> 1
ttreplay:             5 -> 0
```

**[Empirically verified on Blackhole]** The isolated benefit of caching only
`scaled`, with baseline correction behavior retained, is:

| Operation | Baseline | Caching only | Cycles removed | Speedup |
|---|---:|---:|---:|---:|
| `fmod` | 2976.242 | 2912.227 | 64.016 | 1.0220x |
| `remainder` | 3328.242 | 3264.227 | 64.016 | 1.0196x |

The final caching-only `remainder` symbol contains 65 encoded instructions,
one `sfpmul`, and five `ttreplay` instructions. No spill loads/stores were
observed in the symbol. The baseline contained 66 encoded instructions and four
`sfpmul` instructions.

**[Unknown]** End-to-end TTNN latency was not measured because the current
checkout lacks a usable built TTNN Python module.

**[Unknown]** Wormhole cycles were not measured.

## 13. Files that should change

Immediate safe patch:

```text
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_remainder.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_remainder.h
```

Test infrastructure should be extended so the divisor and reciprocal are not
fixed to `2.0` and `0.5`:

```text
tt_metal/tt-llk/tests/helpers/include/sfpu_operations.h
tt_metal/tt-llk/tests/python_tests/test_sfpu_unary.py
tests/ttnn/unit_tests/operations/eltwise/test_div_ops.py
```

## 14. Regression tests to add

At minimum, preserve these exact cases:

```text
(a=50331656,  b=3)
(a=83886104,  b=5)
(a=116040560, b=7)
(a=167772208, b=10)
(a=50855936 BF16-exact, b=3)
```

For each case, capture raw bits for:

```text
rho
scaled
q_hat
residual before correction
residual after positive correction
residual after negative correction
final signed result
```

The test must compare against exact rational quotient/remainder semantics before
format-specific output conversion.

## 15. Final evidence status

| Question | Status |
|---|---|
| Are two corrections globally sufficient? | **No — Proven and host counterexample** |
| Is `|q_hat-q| <= 1` global? | **No — host cases with `k=2` and much larger** |
| Does BF16 input make it safe? | **No — host BF16-exact counterexample** |
| Is the old ten-block reducer globally correct? | **No — Proven** |
| Is caching `scaled` safe? | **Yes — Proven; measured on Blackhole at 1.0196x-1.0220x** |
| Are adversarial bits confirmed on Blackhole? | **Unknown** |
| Are they confirmed on Wormhole? | **Unknown** |
| Is end-to-end TTNN speed measured? | **Unknown** |
