# Research: Hybrid Scalar SFPI `fmod` and Floor Remainder

## 1. Executive conclusion

**[Proven]** A reciprocal/truncation fast path followed by one correction in
each direction is correct only when the quotient-estimation error is at most
one.

**[Proven]** For finite normal binary32 reciprocal and multiplication under
round-to-nearest-even, `exponent(scaled) < 22` is a conservative, cheap
fast-path classifier. It is deliberately stricter than an unqualified `2^23`
ratio threshold.

**[Host verified]** The classifier produced no quotient errors greater than one
over 110 deterministic adversarial cases, an exhaustive sweep of positive finite
BF16 dividends for eight selected divisors (258,348 pairs), and 1,000,000 seeded
random normal-FP32/divisor pairs.

**[Proven for normal IEEE subtraction]** Radix-2 exponent-scaled reduction is a
viable robust magnitude fallback. Its subtraction steps satisfy Sterbenz's
lemma and therefore do not lose low remainder bits merely because `a/b` is
large.

**[Host verified]** The prototype radix-2 fallback exactly matched rational
remainders for every representable normal-domain case above.

**[Blackhole verified]** A minimal raw-bit diagnostic reproduced the deterministic
`k=+2` counterexample on a Blackhole p150b. The classifier marked it unsafe, and
the test-only radix-2 hybrid produced bit-exact fmod and floor-remainder results
for the selected magnitude-3 adversarial set with both divisor signs.

**[Blackhole verified]** The always-present masked radix-2 fallback is not a
viable common-case optimization. Generated code executes a 254-iteration scalar
loop inside each of 32 vector iterations even when all lanes are masked. It
measured 91,229.445 cycles/tile for fmod and 91,901.445 for remainder: 31.33x
and 28.15x slower than the caching-only kernels.

**[Host and Blackhole verified in the stated normal domain]** A follow-up
fixed-stage chunked reducer was derived with a one-sided local quotient and
split divisor products that fit the documented 28-bit partially fused SFPMAD
precision. All five requested chunk sizes passed the proof-oriented host suite;
`CHUNK_BITS=16` was selected and passed 28 Blackhole correctness cases.

**[Rejected architecture]** The standalone C16 robust lower bound measured
37,979.469 cycles/tile for fmod and 38,683.461 for remainder for every tested
0/1/16/32-unsafe-lane mix. It is 13.04x and 11.85x slower than caching-only and
exceeds the predefined 10k rejection threshold. The combined fast/chunked
wrapper also exceeds `L0-L7` and triggers an illegal compiler spill.

**[Compile verified only]** The diagnostic/hybrid sources compile for Wormhole;
the final chunked correctness/performance matrix passed all 36 compile-producer
cases. No Wormhole hardware result is available.

**[Architecture B follow-up, exhaustively verified fixed-divisor domains]** A
scalar-specialized robust fixed schedule removes the dynamic shift extraction
and replaces the general FP32 truncation with `FP32 -> UINT16 -> FP32`. An exact
exponent-127 pre-reduction closes the observed maximum-range product overflow.
All 2,130,706,432 positive normal dividends passed for each of nine
representative moderate divisors; the Blackhole functional/diagnostic runtime
matrix passed 32 cases, both architecture compile matrices passed 40 cases,
and performance is 6171.445 fmod / 6875.445 remainder cycles per tile. This
passes the weak 10k robust gate and is 6.154x/5.626x faster than the C16 lower
bound. FTZ/subnormal, reciprocal-range, and arbitrary-divisor proof boundaries
remain.

**Recommendation:** stop optimizing always-present masked hybrids. Retain the
independently safe caching-only change and continue Architecture B as separate
`FastAssumeBounded` and scalar-specialized fixed-schedule `Robust` compiled
variants. Do not add an automatic tensor scan or integrate the robust candidate
until its documented exclusions are closed. The fixed-schedule result is in
`SFPI_SCALAR_MODULO_FIXED_SCHEDULE_RESEARCH.md`.

## 2. Exact operation contracts

**[Proven from repository implementation and golden tests]** Scalar `fmod` is:

```text
fmod(x,y) = x - trunc(x/y)*y
```

and its nonzero result follows the sign of `x`.

**[Proven from repository implementation and golden tests]** Scalar
`remainder` is Python/PyTorch floor remainder:

```text
remainder(x,y) = x - floor(x/y)*y
```

and its nonzero result follows the sign of `y`. It is not C++
`std::remainder`.

Both operations can first compute a positive magnitude remainder for
`a=abs(x)`, `b=abs(y)`, then apply different sign policies.

## 3. Why the reciprocal algorithm fails globally

Let

```text
t = a/b
q = floor(t)
r = a-q*b, 0 <= r < b
rho = fl32(1/b)
scaled = fl32(a*rho)
q_hat = trunc(scaled)
k = q_hat-q.
```

In exact residual arithmetic:

```text
r_hat = r-k*b.
```

One `-b` and one `+b` correction recover `r` for all residuals exactly when
`k` is `-1`, `0`, or `1`. For `|k|>=2`, one correction leaves the result outside
`[0,b)`.

The deterministic `a=50331656`, `b=3` case has `k=2`. Additional host cases
exist for `5`, `7`, `10`, binary32 `0.1`, `0.3`, and `0.003`, as documented in
`SFPI_SCALAR_MODULO_CORRECTNESS_AUDIT.md`.

## 4. Mathematical fast-path condition

For normal correctly rounded binary32 operations, let unit roundoff be

```text
u = 2^-24.
```

Write

```text
rho = (1/b)(1+delta_1), |delta_1| <= u
scaled = (a*rho)(1+delta_2), |delta_2| <= u.
```

Then

```text
scaled = t(1+epsilon)
|epsilon| <= gamma = 2u+u^2.
```

If the unbiased binary exponent of the nonzero `scaled` value is below 22,
the largest possible encoded value is below `2^22`. Accounting for the inverse
relative-error bound keeps the absolute difference `|scaled-t|` below one.
Therefore truncating `scaled` can differ from `floor(t)` by at most one.

**[Proven under stated assumptions]** This makes

```text
scaled == 0 or exexp(scaled) < 22
```

a sufficient fast condition for finite normal `a`, `b`, `rho`, and `scaled`.

It is conservative. The experimentally safe region is larger for many
mantissas and exact reciprocals, but that observation is not used as proof.

### Exclusions

The proof does not cover:

- reciprocal overflow;
- multiplication overflow;
- subnormal inputs/results under SFPU FTZ;
- NaN or infinity;
- exact output-bit equivalence after partially fused residual formation.

Those cases must be classified separately before the fast path.

## 5. Classifier cost

### A. Exponent of `scaled`

The current quotient extraction already executes `exexp(scaled)`. Reusing it
requires an integer comparison against 22 and predicate state already needed by
the quotient path.

| Property | Assessment |
|---|---|
| Extra exponent extraction | none |
| Extra comparison | approximately one |
| Lane-local | yes |
| Register pressure | minimal |
| Proof quality | strong for normal domain |

This is the recommended mathematical classifier.

### B. Exponent difference `exp(a)-exp(b)`

The divisor exponent can be precomputed by the host, but extracting `exp(a)`
adds an instruction and a live integer vector. A conservative significand-free
bound is possible, but it rejects more lanes than the `scaled` classifier.

### C. Mantissa inspection

Mantissa-aware bounds can enlarge the fast region and recognize power-of-two
divisors. They require more bit operations, constants, and review complexity.
Host specialization is preferable because the divisor is scalar.

### D. Residual postcondition

Checks such as `r<-b` or `r>=2b` detect some failures but cannot prove
correctness: an incorrect approximate quotient can accidentally produce a
residual inside `[0,b)`. Residual checks are defense-in-depth only.

## 6. Fallback algorithms considered

### Candidate A: radix-2 exponent-scaled reduction

Algorithm:

```text
r = a
d = b scaled to the largest power-of-two exponent with d <= r

while d >= b:
    if r >= d:
        r -= d
    d *= 0.5
```

At each step, the invariant before subtraction is `r < 2d`. If `r>=d`, then
`r/d` lies in `[1,2)`. By Sterbenz's lemma, `r-d` is exact in IEEE floating
arithmetic when both values/results remain representable in the relevant
domain. After the subtraction, `r<d`; halving `d` restores `r<2d` for the next
step.

**[Proven for positive normal IEEE values excluding subnormal output]** The
algorithm ends with `0<=r<b` and preserves the exact binary remainder.

**[Host verified]** The FP32 prototype had zero failures in the tested normal
domain.

Worst-case cost is proportional to the exponent gap, up to roughly 253 binary
steps. It is the simplest proof-quality fallback but potentially too slow.

### Candidate B: multi-stage reciprocal chunks

Restricting each quotient chunk to a proven exponent range can reduce iterations.
However, subtracting `q_chunk*d` must preserve low bits, and a partially fused
MAD proof is more complicated than Sterbenz subtraction. The follow-up solved
this in the stated normal domain by using a one-sided quotient estimate and
split divisor components that keep every product within the documented 28-bit
SFPMAD precision. Its C16 device prototype is nevertheless rejected for the
always-present hybrid because the standalone robust lower bound costs about 38k
cycles/tile and the combined wrapper spills.

### Candidate C: radix-4/radix-16 leading digits

Higher radix reduces loop count but needs several divisor multiples and either
multiple comparisons or digit extraction. This path was not pursued after the
shorter proven C16 chunked candidate still failed the performance gate; radix-16
also likely creates excessive local-register and predicate pressure.

### Candidate D: Newton reciprocal refinement

One step

```text
rho1 = rho*(2-b*rho)
```

reduces reciprocal approximation error in favorable cases. It does not solve
the loss of integer quotient bits above `2^24`, even with a perfectly rounded
FP32 reciprocal. It is rejected as a standalone fallback.

### Candidate E: existing primitives

Repository inspection found useful `exexp`, `setexp`, `setman`, `divp2`, shift,
and conversion helpers, plus integer division/remainder implementations. No
native floating modulo instruction or already-proven arbitrary-exponent FP32
remainder reducer was found.

### Candidate F: host-assisted scalar specialization

The host can precompute:

- biased exponent of `b`;
- normalized mantissa of `b`;
- whether `b` and `1/b` are exact powers of two;
- constants for radix-4 multiples.

This reduces per-vector setup. It cannot classify `a/b` completely because `a`
varies per lane.

## 7. Correctness proof status

| Component | Status |
|---|---|
| Two corrections when `|k|<=1` | **Proven** |
| `exexp(scaled)<22` classifier, normal RNE domain | **Proven** |
| Radix-2 reduction, normal IEEE domain | **Proven** |
| Radix-2 SFPU bit-exact behavior | **Blackhole verified for selected normal magnitude-3 cases and both divisor signs; general domain unknown** |
| C16 chunked reduction, normal non-FTZ proof domain | **Proven and host verified; selected Blackhole cases pass** |
| Subnormal/FTZ fallback behavior | **Unknown** |
| Radix-4 fallback | **Unknown** |
| Full hybrid special-value behavior | **Unknown** |

## 8. Rejected candidates

- Unconditional two corrections: deterministic `k=2` counterexamples.
- Existing ten blocks: `b-r` is not modulo subtraction and negative residuals
  are not repaired.
- Threshold chosen only from tests near `2^23`: no proof.
- Residual-only classifier: cannot detect every wrong in-range residual.
- Reciprocal refinement alone: quotient integer-representability remains.
- Naive repeated `r-=b`: for large exponent gaps, subtraction can round back to
  `r` and make no progress.

## 9. Host exact-model results

The reusable framework is:

```text
tools/sfpi_modulo_hybrid_reference.py
```

It represents input FP32 values as exact `Fraction` objects for the reference
quotient/remainder, while executing the candidate path with binary32 values.

Latest seeded results:

```text
deterministic cases=110 unsafe=68 fallback_failures=0
BF16 exhaustive safe=151180 fallback=107168
random safe=581735 fallback=411380 seed=0x5f91
```

The BF16 run exhausts all positive finite BF16 bit patterns for eight selected
divisors. The random run samples one million normal FP32 dividends. This is not
an exhaustive search over all FP32 pairs.

## 10. Blackhole diagnostic results

**[Blackhole verified]** A minimal diagnostic ran on Blackhole p150b with the
entire input face set to `a=50331656`. It checked these exact FP32 bit patterns:

```text
a                            0x4c400002
b                            0x40400000
rho                          0x3eaaaaab
scaled                       0x4b800002
q_hat                        0x4b800002
residual before correction   0xc0800000  (-4)
after positive correction    0xc0800000  (-4)
after negative correction    0xbf800000  (-1)
safe classifier              0x00000000  (false)
```

The initial monolithic diagnostic also attempted to retain fallback checkpoints.
It exceeded the eight local SFPU registers and triggered the compiler error
`cannot store sfpu register (register spill)`. Splitting the raw-bit diagnostic
from fallback validation removed the spill.

**[Blackhole verified for the tested set]** The separate hybrid tests matched
PyTorch bit-for-bit for positive/negative safe values, both scalar signs, the
FP32 and BF16-exact large magnitude-3 counterexamples, values adjacent to
`3*2^24`, exact multiples, fmod, and floor remainder. The remainder prototype
needed sign-mismatch adjustment for nonzero results and explicit dividend-sign
restoration for exact zero.

## 11. Wormhole diagnostic results

**[Compile verified only]** All three diagnostic/hybrid test variants compile
with `-mcpu=tt-wh-tensix`. No Wormhole hardware is available, so numerical and
performance behavior remain unknown. Wormhole also differs from Blackhole in
documented negative-zero and SFPMAD scheduling details.

## 12. Generated-disassembly comparison

Measured Blackhole baseline versus final caching-only implementation:

| Metric | Baseline | Caching only |
|---|---:|---:|
| Encoded instructions | 66 | 65 |
| `sfpmul` | 4 | 1 |
| `ttreplay` | 5 | 5 |
| Observed spills | none | none |

The experimental hybrid disassembly has a compact static body only because the
fallback is a runtime scalar loop:

| Architecture | fmod symbol | remainder symbol | Fallback stage body |
|---|---:|---:|---|
| Blackhole | 70 encoded instructions | 91 | 10 SFPU + 2 scalar instructions |
| Wormhole compile-only | 79 encoded instructions | 100 | 11 SFPU + 2 scalar instructions |

The stage body executes 254 times inside every one of the 32 vector iterations.
Blackhole uses all local registers `L0` through `L7` but does not spill. Wormhole
adds an `sfpnop` in the stage body for its documented post-MAD scheduling rule.

## 13. Register-pressure and vector-dispatch analysis

An exponent-scaled fallback needs at least live `r`, `d`, `b`, exponent/counter,
and predicate state in addition to sign values. Radix-4 needs more divisor
multiples or temporary digit state.

More importantly, SFPI `v_if` is lane predication. Predicated vector
instructions still occupy the instruction stream. The generated disassembly
confirms that the scalar 254-stage loop is outside any scalar early exit.
Consequently:

- per-lane fallback preserves correctness but not necessarily common-case
  latency;
- one unsafe lane can cost nearly the same as all lanes unsafe;
- staged masked fallback avoids corrupting safe lanes but does not make its
  instructions free;
- whole-vector dispatch needs an efficient any-lane reduction/branch mechanism,
  which has not been identified in the current SFPI path.

This is now a measured rejection of the always-present radix-2 architecture,
not merely a risk.

## 14. Performance tables

Blackhole MATH_ISOLATE, FP32 input/output, destination accumulation enabled:

| Variant | `fmod` cycles/tile | `remainder` cycles/tile | Correctness status |
|---|---:|---:|---|
| Original baseline | 2976.242 | 3328.242 | known incomplete reducer |
| Caching only | 2912.227 | 3264.227 | preserves baseline behavior |
| Two corrections | 1568.227 | 1888.227 | rejected globally |
| Masked radix-2 hybrid | 91229.445 | 91901.445 | selected normal cases pass; architecture rejected for performance |
| C16 robust lower bound | 37979.469 | 38683.461 | selected normal cases pass; single-kernel architecture rejected |

The masked hybrid is 31.33x slower than caching-only fmod and 28.15x slower than
caching-only remainder. The MATH_ISOLATE probe used the same emitted kernel for
all lane data; disassembly proves that changing the number of unsafe lanes does
not change the 254-stage scalar loop issue count. A separate 0/1/16/32-lane
full-pipeline sweep was not completed.

End-to-end TTNN and Wormhole measurements remain **[Unknown]**.

## 15. Recommended hybrid algorithm

If device validation confirms the primitives, the preferred correctness design
is:

```text
classify special/overflow/FTZ cases
compute cached scaled and its exponent

safe lane: exexp(scaled)<22
    reciprocal quotient
    partially fused residual
    one correction each direction

unsafe lane:
    radix-2 or proven radix-4 exponent-scaled reduction

common operation-specific sign restoration
```

The exact algorithm remains a correctness reference, but neither measured
masked implementation may be used as an always-present production path. The
radix-2 version costs about 91k cycles/tile and the shorter C16 lower bound still
costs about 38k. The single-kernel architecture is rejected.

The next design should use two separately compiled internal variants:

- `FastAssumeBounded`: no classifier or fallback; the caller has already proven
  every lane satisfies the normal bounded-range contract;
- `Robust`: the generic default for unknown ranges, using a correctness-first
  range reducer.

Do not add an automatic tensor scan. Selection should come only from an
explicit internal guarantee or existing planner/range metadata. Questions for
the SFPI/LLK developer remain:

- a true vector-wide early exit before the fallback loop;
- the supported form for two compiled variants and range metadata;
- a portable way to retain low product bits and avoid SFPU register spills.

The scalar `b` alone cannot select a safe kernel because `a` varies per lane.

## 16. Production files eventually affected

No hybrid production edit is justified yet. A validated implementation would
touch:

```text
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_remainder.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_remainder.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
```

The host generator change is needed only if extra scalar-specialization
constants or variants are selected.

## 17. Regression tests to add

- Parameterize LLK scalar divisor/reciprocal instead of fixing `2.0/0.5`.
- Keep the completed exact `k=2` Blackhole raw-bit diagnostics for divisors
  `3`, `5`, `7`, and `10`.
- Keep the completed BF16-exact `a=50855936`, `b=3` case.
- Sweep exact multiples and neighboring FP32 values around quotient exponents
  20 through 30.
- Retain the completed BF16 host sweeps for representative scalar divisors.
- Retain the completed signed `fmod` and floor-remainder combinations for the
  selected device cases.
- Add NaN, infinity, signed zero, normal/subnormal boundary, reciprocal overflow,
  and scaled overflow cases according to the documented SFPU contract.
- If a new fallback avoids the unconditional scalar loop, benchmark safe-only,
  one unsafe lane, half unsafe, and all-unsafe tiles.

## 18. Remaining unknowns

1. Bit-exact agreement outside the selected Blackhole normal cases for divisors
   `3`, `5`, `7`, `10`, FP32 `0.1`, and FP32 `0.3`, especially FTZ edges.
2. A correct policy for subnormal divisor/remainder under the existing API.
3. Efficient vector-wide early exit or separate-kernel dispatch on Tensix.
4. Closing the fixed schedule's FTZ/subnormal and top-range product-overflow
   exclusions; performance is now below the weak 10k robust gate.
5. Wormhole runtime correctness and performance.
6. End-to-end TTNN latency and real workload unsafe-lane frequency.

Until these are resolved, only caching the repeated reciprocal product should
be considered production-safe.

## 19. Fixed-stage chunked follow-up

The completed follow-up is documented in
`SFPI_SCALAR_MODULO_CHUNKED_RESEARCH.md`. In brief:

- `CHUNK_BITS={8,12,16,18,20}` all had zero failures inside the explicit host
  proof domain over 582 deterministic/boundary pairs, 357,632 BF16/divisor
  pairs, and one million seeded normal-FP32 pairs per candidate;
- the local quotient error was always `{0,+1}`, one correction sufficed, and
  nonterminal exponent progress was at least `CHUNK_BITS-1`;
- `CHUNK_BITS=16` minimizes fixed split-component subtractions (34) while using
  17 stages and two at-most-12-bit divisor components;
- 8,525 pairs were excluded because an exact intermediate/result became
  subnormal and entered unresolved SFPU FTZ behavior;
- strict Blackhole raw-bit diagnostics passed for divisors `3`, `5`, `7`, and
  `10`; functional tests also covered FP32 `0.1` and `0.3` to exercise a nonzero
  low divisor component;
- the standalone robust candidate passed 28 Blackhole cases, while the combined
  cached-fast/chunked wrapper caused an illegal SFPU spill;
- the lower bound was identical for 0/1/16/32 unsafe lanes at 37,979.469
  fmod cycles/tile and 38,683.461 remainder cycles/tile, so architecture A was
  rejected and architecture B was specified.

## 20. Architecture B robust fixed schedule

The first Architecture B robust prototype is documented in
`SFPI_SCALAR_MODULO_FIXED_SCHEDULE_RESEARCH.md`. It specializes the shift
sequence and split divisor for the scalar at compile time:

```text
K0=max(112-Eb,0), then K0,K0-15,...,0.
```

This bounds the local quotient below `2^16` and reduces the Blackhole repeated
stage from 66 SFPU instructions to 14 SFPU plus two scalar loop instructions.
An integer-lattice verifier exhaustively accepted every positive normal FP32
dividend for nine representative moderate scalar divisors. The exact
exponent-127 pre-reduction is also covered by the Blackhole raw-bit matrix.
Generated code has no spills, and Blackhole MATH_ISOLATE is
6171.445/6875.445 cycles per tile. This keeps the Architecture B assessment at
“performance-viable research prototype,” not production-ready: measured FTZ
behavior, reciprocal/special-value policy, and arbitrary-divisor proof remain,
and Wormhole evidence is still compile-only.
