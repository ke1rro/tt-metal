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
over 110 deterministic adversarial cases, all 258,348 tested BF16/divisor pairs,
and 1,000,000 seeded random normal-FP32 pairs.

**[Proven for normal IEEE subtraction]** Radix-2 exponent-scaled reduction is a
viable robust magnitude fallback. Its subtraction steps satisfy Sterbenz's
lemma and therefore do not lose low remainder bits merely because `a/b` is
large.

**[Host verified]** The prototype radix-2 fallback exactly matched rational
remainders for every representable normal-domain case above.

**[Inferred]** A same-kernel per-lane hybrid is unlikely to retain the full fast
path speedup. SFPU vector conditionals mask lanes but do not turn a long fallback
instruction stream into a free branch. One unsafe lane can make the vector issue
the same fallback stream as 32 unsafe lanes.

**[Unknown]** The classifier, fallback checkpoints, and adversarial result have
not yet been captured as raw bits by a dedicated Blackhole diagnostic kernel.
Wormhole hardware is unavailable.

**Recommendation:** do not merge the hybrid yet. Retain the independently safe
caching-only change, add an on-device diagnostic, then prototype radix-2 and a
chunked-radix fallback as separate experimental kernels. The central unresolved
question is vector dispatch cost, not host mathematical feasibility.

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
MAD proof is more complicated than Sterbenz subtraction. This remains a useful
prototype candidate, not a proven fallback.

### Candidate C: radix-4/radix-16 leading digits

Higher radix reduces loop count but needs several divisor multiples and either
multiple comparisons or digit extraction. Radix-4 is the next candidate to
prototype; radix-16 likely creates excessive local-register and predicate
pressure.

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
| Radix-2 SFPU bit-exact behavior | **Unknown** pending device/model validation |
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

**[Unknown]** A raw-bit intermediate diagnostic has not yet been completed.

Existing Blackhole LLK final-output tests pass only with scalar `2.0`, whose
reciprocal `0.5` is exact. They do not validate the classifier or fallback.

Required diagnostic outputs are `a`, `b`, `rho`, `scaled`, `q_hat`, candidate
residual, classifier, fallback checkpoints, final magnitude, and signed result.

## 11. Wormhole diagnostic results

**[Unknown]** No Wormhole hardware is available. Compile-time similarity cannot
be reported as runtime equivalence. Wormhole also differs from Blackhole in
documented negative-zero and SFPMAD scheduling details.

## 12. Generated-disassembly comparison

Measured Blackhole baseline versus final caching-only implementation:

| Metric | Baseline | Caching only |
|---|---:|---:|
| Encoded instructions | 66 | 65 |
| `sfpmul` | 4 | 1 |
| `ttreplay` | 5 | 5 |
| Observed spills | none | none |

No hybrid fallback disassembly is reported because no candidate has passed the
required on-device validation gate.

## 13. Register-pressure and vector-dispatch analysis

An exponent-scaled fallback needs at least live `r`, `d`, `b`, exponent/counter,
and predicate state in addition to sign values. Radix-4 needs more divisor
multiples or temporary digit state.

More importantly, SFPI `v_if` is lane predication. Predicated vector
instructions still occupy the instruction stream. Consequently:

- per-lane fallback preserves correctness but not necessarily common-case
  latency;
- one unsafe lane can cost nearly the same as all lanes unsafe;
- staged masked fallback avoids corrupting safe lanes but does not make its
  instructions free;
- whole-vector dispatch needs an efficient any-lane reduction/branch mechanism,
  which has not been identified in the current SFPI path.

This is the primary risk to the proposed hybrid architecture.

## 14. Performance tables

Blackhole MATH_ISOLATE, FP32 input/output, destination accumulation enabled:

| Variant | `fmod` cycles/tile | `remainder` cycles/tile | Correctness status |
|---|---:|---:|---|
| Original baseline | 2976.242 | 3328.242 | known incomplete reducer |
| Caching only | 2912.227 | 3264.227 | preserves baseline behavior |
| Two corrections | 1568.227 | 1888.227 | rejected globally |

Hybrid performance is **[Unknown]** pending an on-device prototype. End-to-end
TTNN and Wormhole measurements are also **[Unknown]**.

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

For performance, first investigate whether a vector-wide early exit is possible.
If it is not, a separate robust kernel selected by an explicit host-visible
input-range contract may outperform an always-present masked fallback. Such a
contract cannot be inferred from scalar `b` alone.

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
- Add the exact `k=2` counterexamples for divisors `3`, `5`, `7`, and `10`.
- Add BF16-exact `a=50855936`, `b=3`.
- Sweep exact multiples and neighboring FP32 values around quotient exponents
  20 through 30.
- Exhaust BF16 inputs for representative scalar divisors in the host model.
- Validate signed `fmod` and floor-remainder combinations separately.
- Add NaN, infinity, signed zero, normal/subnormal boundary, reciprocal overflow,
  and scaled overflow cases according to the documented SFPU contract.
- Capture raw intermediate device bits for deterministic adversarial cases.
- Benchmark safe-only, one unsafe lane, half unsafe, and all-unsafe tiles.

## 18. Remaining unknowns

1. Bit-exact agreement between host radix-2 subtraction and partially fused/
   FTZ SFPU behavior.
2. A correct policy for subnormal divisor/remainder under the existing API.
3. Efficient vector-wide early exit or dispatch on Tensix SFPU.
4. Radix-2 versus radix-4 device instruction count and register pressure.
5. Blackhole adversarial intermediate bit patterns.
6. Wormhole correctness and performance.
7. End-to-end TTNN latency and real workload unsafe-lane frequency.

Until these are resolved, only caching the repeated reciprocal product should
be considered production-safe.
