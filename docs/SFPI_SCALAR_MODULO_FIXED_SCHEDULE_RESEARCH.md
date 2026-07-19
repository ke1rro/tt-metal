# SFPI scalar modulo fixed-schedule robust research

## 1. Result

Architecture B now has a test-only scalar-specialized candidate that is exact
for every positive normal FP32 dividend for nine representative normal scalar
divisors.  The exhaustive verifier processed `2,130,706,432` inputs per divisor
and found no proof-domain exclusion or final mismatch for:
xq
```text
3, 5, 7, 10, FP32 0.1, FP32 0.3, 8,
nextDown(8), nextUp(8)
```

This is `19,176,357,888` fully accepted input/divisor cases.  It is not a proof
for every FP32 divisor: SFPU FTZ behavior is observable for very small normal
divisors, a very large divisor can have a subnormal reciprocal, and the public
operator policy for subnormal/special inputs remains to be defined.

For scalar divisor `3.0f`, Blackhole p150b MATH_ISOLATE measured:

| Operation | Fixed schedule | C16 dynamic schedule | Speedup | Caching-only reference |
|---|---:|---:|---:|---:|
| `fmod` | 6171.445 cycles/tile | 37979.469 | 6.154x | 2912.227 |
| floor `remainder` | 6875.445 cycles/tile | 38683.461 | 5.626x | 3264.227 |

The hardened kernel is 2.119x/2.106x slower than the caching-only incomplete
reducer, but remains below the predefined `>10000 cycles/tile` rejection gate.
Production headers were not modified.

## 2. Fixed schedule and local proof

For positive normal scalar divisor `b`, with unbiased exponent `Eb`:

```text
K0 = max(112-Eb, 0)
schedule = K0, K0-15, K0-30, ..., 0
d(K) = b * 2^K
```

The first local ratio is below `2^16`.  After a stage the residual is in
`[0,d)`, and reducing the next scaled divisor by `2^15` bounds every later local
ratio below `2^15`.  There are at most 17 stages over the normal-reciprocal
domain; all nine exhaustively tested moderate divisors use nine stages.

The upward reciprocal is:

```text
rho_up = nextUp(nextUp(RN32(1/b)))
```

An exact power-of-two divisor retains its exact reciprocal.  The proven local
bound is `t <= z < t + 0.376`, so nearest `FP32 -> UINT16 -> FP32` conversion
produces `q_hat` in `{floor(t), floor(t)+1}`.  Splitting the scaled divisor into
two components of at most 12 significand bits keeps each 16b-by-12b product in
the documented 28-bit SFPMAD partial-product precision.  High-to-low fused
subtractions followed by one negative correction therefore recover the exact
nonnegative residue when all required transients are normal or zero.

## 3. Exact maximum-exponent pre-reduction

The initial implementation exposed a narrow top-range product overflow:

| Divisor | Excluded dividend interval | Count |
|---|---:|---:|
| `3` | `0x7f7fffbc..0x7f7fffff` | 68 |
| `5` | `0x7f7fff9d..0x7f7fffff` | 99 |
| `10` | `0x7f7fff9d..0x7f7fffff` | 99 |

`7`, FP32 `0.1`, FP32 `0.3`, `8`, and the two neighbors of `8` had no such
case.  The overflow is now removed with an exact special case for exponent-127
dividends:

```text
h = a / 2                         # exact and normal
r = fixed_schedule_mod(h, b)
a mod b = (2*r) mod b             # at most one subtraction of b
```

This keeps the first split partial product below FP32 overflow.  The C++
verifier and the SFPI test kernel implement the same transformation.  With it,
all nine complete positive-normal sweeps pass without exclusions.

## 4. Exhaustive verifier

The optimized verifier is:

```text
tools/sfpi_modulo_fixed_schedule_exhaustive.cpp
```

It uses a fixed-divisor integer-lattice oracle based on significands, exponents,
and modular powers of two; neither `std::fmod` nor host floating-point remainder
is used as the reference.  It separately classifies quotient errors, split
component precision, partial-product overflow, intermediate precision,
subnormal/FTZ transients, correction errors, and final mismatches.  The input
space can be sharded and processed by multiple threads.

Example:

```bash
g++ -std=c++20 -O3 -march=native -DNDEBUG -ffp-contract=off -pthread \
    -Wall -Wextra -Werror tools/sfpi_modulo_fixed_schedule_exhaustive.cpp \
    -o /tmp/sfpi_modulo_fixed_schedule_exhaustive

/tmp/sfpi_modulo_fixed_schedule_exhaustive \
    --divisor-bits 0x40400000 --exhaustive-inputs \
    --prehalve-max-exponent --threads 16
```

Full positive-normal results with pre-halving:

| Divisor bits | Value | Passed | Classified exclusions | Mismatches |
|---:|---:|---:|---:|---:|
| `0x40400000` | 3 | 2,130,706,432 | 0 | 0 |
| `0x40a00000` | 5 | 2,130,706,432 | 0 | 0 |
| `0x40e00000` | 7 | 2,130,706,432 | 0 | 0 |
| `0x41200000` | 10 | 2,130,706,432 | 0 | 0 |
| `0x3dcccccd` | FP32 0.1 | 2,130,706,432 | 0 | 0 |
| `0x3e99999a` | FP32 0.3 | 2,130,706,432 | 0 | 0 |
| `0x41000000` | 8 | 2,130,706,432 | 0 | 0 |
| `0x40ffffff` | `nextDown(8)` | 2,130,706,432 | 0 | 0 |
| `0x41000001` | `nextUp(8)` | 2,130,706,432 | 0 | 0 |
| `0x00800000` | smallest normal | 1,946,157,055 | 184,549,377 subnormal transients | 0 |

Including the smallest-normal-divisor run, the verifier processed
`21,307,064,320` normal input/divisor cases with no proof or final mismatch.
The final row is not counted as a correctness pass: its 184,549,377
`IntermediateSubnormal` cases are explicitly outside the normal-transient
proof domain.  The largest finite normal divisor is rejected at configuration
time because its reciprocal is subnormal.

Additional complete classification sweeps for divisor `3` found:

```text
positive zero:             1 Passed
positive subnormal inputs: 8,388,607 InputSubnormal
positive Inf/NaN patterns: 8,388,608 UnsupportedSpecialValue
```

Four-way sharding reproduced the same histograms as the unsharded top-range
run.  A 4096-input top-range slice also matched the independent Python exact
model classification-for-classification.

## 5. Independent Python model

The slower proof-oriented model is:

```text
tools/sfpi_modulo_fixed_schedule_reference.py
```

Selected deterministic, BF16, and 100k random normal pairs:

```bash
python3 tools/sfpi_modulo_fixed_schedule_reference.py \
    --random 100000 --prehalve-max-exponent
```

```text
deterministic_and_boundaries=582
bf16_positive_finite_pairs=357632
random_normal_pairs=100000 divisor_mode=selected seed=0x5f91
fixed_c16 tested=456700 failures=0 exclusions=1514
schedule_stages=17 active_stages=9 q_error=[0,1] max_q=32768 prehalved=1842
```

The remaining exclusions are 1511 exact subnormal intermediates/results and
three subnormal component subtractions.  The former three top-range overflow
cases now pass.

An arbitrary-normal-divisor sample gives:

```bash
python3 tools/sfpi_modulo_fixed_schedule_reference.py \
    --random 100000 --skip-bf16 --random-divisors --seed 0x7a11 \
    --prehalve-max-exponent
```

```text
fixed_c16 tested=95002 failures=0 exclusions=5580
schedule_stages=17 active_stages=17 q_error=[0,1] max_q=32768 prehalved=366
```

| Exclusion | Count |
|---|---:|
| split divisor component becomes subnormal | 4063 |
| biased reciprocal is not normal | 791 |
| exact result/intermediate is subnormal | 428 |
| component subtraction becomes subnormal | 298 |

No partial-product or post-reduction overflow remains in that sample.

## 6. SFPI implementation and device correctness

All implementation changes remain test-only:

```text
tt_metal/tt-llk/tests/helpers/include/scalar_modulo_fixed_schedule_research.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_fixed_schedule_test.cpp
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_fixed_schedule_perf.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_fixed_schedule.py
tt_metal/tt-llk/tests/python_tests/perf_sfpu_scalar_modulo_fixed_schedule.py
```

The strict functional matrix covers fmod and floor remainder, both scalar
signs, and divisors `3`, `5`, `7`, `10`, FP32 `0.1`, FP32 `0.3`, and `8`.
Each case contains boundaries, known counterexamples, 231 seeded normals, and
eight exponent-127 values around the former overflow intervals.  Expected bits
come from a Python integer-lattice oracle with exact round-to-nearest-even
composition; PyTorch is intentionally not the oracle because its CPU fmod can
produce NaN for finite top-range inputs such as `FLT_MAX % FP32(0.1)`.

Results:

```text
Blackhole functional/diagnostic runtime: 32 passed
Blackhole performance runtime:            8 passed
Blackhole compile matrix:                 40 passed
Wormhole compile matrix:                  40 passed
```

Wormhole evidence is compile/disassembly only; no Wormhole runtime result is
claimed.

## 7. Measured FTZ behavior

For divisor `0x00800000` (the smallest normal), a Blackhole observation suite
recorded:

| Input | Exact remainder | Blackhole result |
|---:|---:|---:|
| `0x00800000` | `0x00000000` | `0x00000000` |
| `0x00800001` | `0x00000001` | `0x00000000` |
| `0x00800002` | `0x00000002` | `0x00000000` |
| `0x00800100` | `0x00000100` | `0x00000000` |
| `0x00ffffff` | `0x007fffff` | `0x00800000` |
| `0x01000000` | `0x00000000` | `0x00000000` |
| `0x01000001` | `0x00000002` | `0x00000000` |

The ISA documentation states that SFPMAD flushes denormal results to zero on
both architectures; Blackhole preserves the sign of the flushed zero.  Raw
final-stage diagnostics agree:

```text
input 0x00800001:
  after high subtraction: positive subnormal -> +0

input 0x00ffffff:
  scaled quotient rounds to 2
  negative subnormal residual -> -0
  negative correction path adds b -> 0x00800000
```

The last line is an inference from the raw stage values and documented FTZ
semantics.  This is a real semantic boundary, not a host-model false positive.

## 8. Generated code and performance

For divisor `3`, the hardened Blackhole symbols are:

| Operation | Symbol bytes | Encoded instructions | SFPU instructions | Loads/stores |
|---|---:|---:|---:|---:|
| `fmod` | 312 | 78 | 70 | 0 |
| remainder | 400 | 100 | 92 | 0 |

Wormhole compile-only symbols are 392 bytes/98 instructions/88 SFPU for fmod
and 480/120/110 for remainder, with no loads/stores.  No spill was observed on
either architecture.

Blackhole p150b MATH_ISOLATE for 0, 1, 16, and 32 active large-ratio lanes:

| Operation | 0 lanes | 1 lane | 16 lanes | 32 lanes | Text bytes |
|---|---:|---:|---:|---:|---:|
| `fmod` | 6171.445 | 6171.445 | 6171.445 | 6171.445 | 2610 |
| remainder | 6875.445 | 6875.445 | 6875.445 | 6875.445 | 2698 |

The exact top-range guard costs 576 cycles/tile relative to the first
fixed-schedule prototype.  Lane-mix invariance is expected because SFPU
predication still issues the fixed instruction stream.

## 9. Decision and next work

Architecture A remains rejected: issuing a cached fast path plus a masked
dynamic fallback paid the fallback cost regardless of active lanes.
Architecture B remains the preferred split:

```text
FastBounded: explicit caller-proven range contract
Robust:      scalar-specialized fixed schedule
```

The top-range overflow blocker is closed for the implemented schedule.  Before
calling the kernel a generic production implementation:

1. Define zero-divisor, special-value, signed-zero, subnormal-input, and FTZ
   result policy at the operator boundary.
2. Choose and prove a strategy for FTZ-domain remainders, likely an integer/raw
   path or an explicit documented semantic restriction.
3. Reduce the arbitrary-divisor exclusion classes, especially subnormal split
   components and reciprocal range.
4. Add raw integer-store diagnostics if exact distinction among zero/subnormal
   encodings is needed during further SFPU investigation.
5. Run the correctness and performance matrices on Wormhole hardware and tune
   its dependency schedule.
6. Add planner/API dispatch only after the `FastBounded` and `Robust` contracts
   are explicit and reviewed.

No production integration or push was performed. This completed milestone was
later saved in local commit `75855e3002` before starting the normalized-frame
follow-up.

## 10. Exponent-stationary follow-up

The subsequent host-only normalized-frame architecture closes the measured FTZ
and reciprocal-range exclusions for its tested domains. It keeps the divisor
at exponent 111, scales only the residual between stages, and uses an exact
integer raw pack for the physical result. All positive normal inputs passed for
the smallest and largest normal divisors, three additional `Eb=-126` boundary
mantissas, and the nine divisors above: `29,829,890,048` accepted cases with no
exclusion or mismatch. See
`SFPI_SCALAR_MODULO_EXPONENT_STATIONARY_RESEARCH.md`. Device implementation and
performance remain untested.
