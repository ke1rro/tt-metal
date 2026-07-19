# SFPI Scalar Modulo Research — Work State

Last updated: 2026-07-19 UTC

This file is a restart/handoff record for the scalar SFPI `fmod` and floor-style
`remainder` optimization research. It intentionally records unfinished work as
well as verified results.

## Repository and Git state

- Checkout: `/home/user/tt-metal`
- Branch: `opt/optimize-sfpi-scalar-modulo`
- HEAD: `94ed31d5e7` (`research: characterize scalar modulo raw-pack transport`)
- Tracking remote: `origin/opt/optimize-sfpi-scalar-modulo`
- Fork: `https://github.com/ke1rro/tt-metal.git`
- Git identity requested by the user:
  - username: `ke1rro`
  - email: `nikitalenyk2@gmail.com`
- Commit `86ba7d8b5e` is a pushed WIP checkpoint containing the reports, exact host
  model, test-only hybrid prototype, and the production caching-only rollback.
- Parent commit `375862277bc` contains the rejected unconditional two-correction
  implementation. Do not present it as correctness-safe.
- The four production headers in HEAD retain the old ten-block correction
  behavior and only cache `scaled = v * recip_val`. Current uncommitted work does
  not modify production headers.
- Commit `94ed31d5e7` is the isolated raw-pack transport milestone. Blackhole
  preserves subnormal FP32 encodings through that path; Wormhole's passing
  architecture contract expects signed-zero FTZ through the tested opaque-32
  store and UInt32 pack path.
- Current uncommitted work adds the test-only Blackhole exponent-stationary
  normalized reducer, its exact raw-bit silicon test, and its device report.
  It does not change production headers or implement sign/final packing.
- `tt-isa-documentation/` is an untracked user-supplied directory. Do not stage,
  edit, or delete it.
- `.agents/` is also untracked and unrelated. Preserve it.

## User goal

Research, mathematically justify, prototype, validate, disassemble, and benchmark
the Architecture A fixed-stage chunked hybrid first:

```text
proven safe quotient range
    -> cached reciprocal product + truncation + two corrections

unsafe/large quotient range
    -> short fixed-stage chunked range reduction
```

If Architecture A misses correctness, compiler, or performance gates, stop it
and specify Architecture B as separate `FastBounded` and `Robust` compiled
kernels without an automatic tensor scan. Architecture A missed the
compiler/performance gates. Architecture B's robust scalar-specialized fixed
schedule has now been prototyped and passes the weak Blackhole `<10k` gate on
the exhaustively verified fixed-divisor normal domains documented below.

The original design brief and the latest review are in:

```text
/home/user/.codex/attachments/85cf7851-d51f-4d10-b3b0-adf43a363e57/pasted-text.txt
/home/user/.codex/attachments/0ee83c9d-fc3b-46f8-a611-3e7e63b697d4/pasted-text.txt
/home/user/.codex/attachments/2a8f6469-4af2-4a92-b90f-83ed323fc2ff/pasted-text.txt
/home/user/.codex/attachments/1fb41107-fe79-4207-9382-2bfe7bea0024/pasted-text.txt
/home/user/.codex/attachments/8912644b-b9b9-4ce6-be3b-a4df6bbf6a16/pasted-text.txt
/home/user/.codex/attachments/1a714ba4-a7f6-4b4b-84ef-a20b00685b18/pasted-text.txt
```

Do not integrate Architecture A into production: its host/device research is
complete enough to reject it on compiler and performance grounds. The raw-pack
milestone now has real Blackhole and Wormhole silicon evidence. The isolated
stationary normalized reducer now also has Blackhole compile, silicon, and
disassembly evidence; physical finalization remains separate.

## Production header state

The following four local files currently contain the caching-only version:

```text
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_remainder.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_remainder.h
```

The safe local optimization is:

```cpp
const sfpi::vFloat scaled = v * recip_val;
sfpi::vInt exp = sfpi::exexp(scaled);
```

and reuse of `scaled` during quotient extraction. The old ten correction blocks
are retained locally for baseline compatibility. They are not proven as a
general modulo reducer; caching-only preserves baseline behavior rather than
fixing that mathematical issue.

## Correctness findings

### Exact semantics

- Scalar `fmod`: `x - trunc(x/y)*y`; a nonzero result follows the sign of `x`.
- Scalar unary `remainder`: Python/PyTorch floor remainder
  `x - floor(x/y)*y`; a nonzero result follows the sign of `y`.
- This `remainder` is not C++ `std::remainder`.

### Why unconditional two corrections fail

Let `q_hat = q + k`, with exact magnitude remainder `r` in `[0,b)`. The
estimated residual is `r - k*b`. One subtraction for `>=b` followed by one
addition for `<0` is sufficient for all `r` exactly when `k` is in
`{-1,0,+1}`.

Deterministic FP32 counterexample:

```text
a = 50331656.0       bits 0x4c400002
b = 3.0              bits 0x40400000
rho = fl32(1/3)      bits 0x3eaaaaab
scaled               bits 0x4b800002 = 16777220.0
exact q              16777218
q_hat                16777220
k                    +2
residual before      -4
after one +b         -1
correct remainder    2
```

Exact host counterexamples were also found for divisors `5`, `7`, `10`, FP32
`0.1`, `0.3`, and `0.003`.

BF16-exact counterexample promoted to FP32:

```text
BF16 bits 0x4c42 -> a = 50855936.0, FP32 bits 0x4c420000
b = 3
q = 16951978
q_hat = 16951980
k = +2
```

The old ten-block expression `if (v >= s) v = s - v` is also not a correct
general range reducer and does not repair negative residuals caused by an
overestimated quotient.

### Proven conservative classifier

For finite normal binary32 reciprocal and multiplication under RNE, with unit
roundoff `u = 2^-24`, combined relative error is bounded by:

```text
gamma <= 2u + u^2
```

Under those explicit assumptions, the sufficient classifier is:

```text
scaled == 0 || exexp(scaled) < 22
```

It keeps the absolute scaled-versus-exact quotient error below one, hence
`|q_hat-q| <= 1`. It reuses the exponent already extracted by quotient
truncation. `exexp(scaled) < 23` was not accepted as proven.

Exclusions still needing policy/device validation: NaN, infinity, reciprocal or
product overflow, subnormal inputs/results, SFPU FTZ effects, and exact
partially-fused residual bits.

### Fallback conclusions

The leading candidate is radix-2 exponent-scaled reduction:

```text
r = a
d = b * 2^k, largest such d <= r
while d >= b:
    if r >= d:
        r -= d
    d *= 0.5
```

At each subtraction, `r/d` is in `[1,2)`, so Sterbenz's lemma makes the
subtraction exact in the normal IEEE domain. The simple proof excludes
subnormal/FTZ behavior. Worst-case work is about 254 binary stages.

Rejected as standalone solutions:

- unconditional two corrections;
- an empirical `2^23` threshold;
- residual-only classification;
- Newton reciprocal refinement alone;
- naive repeated subtraction by small `b`;
- the existing ten `b-r` blocks as a correctness proof.

The subsequent fixed-stage chunked investigation found a proof-safe way to use
16 quotient bits per stage, but its measured cost still rejects an
always-present hybrid. The selected normal-domain candidate uses:

- `CHUNK_BITS=16`, hence at most `ceil(253/15)=17` stages;
- a two-ULP one-sided upper reciprocal;
- a quotient estimate proven to be either exact or one too high;
- a split divisor with two components of at most 12 significant bits each, so
  every quotient-component product fits the SFPMAD 28-bit partial-fusion limit;
- one negative correction per stage.

This candidate is correctness-promising but not a production result. Its proof
excludes FTZ/subnormal and special-value behavior, and its Blackhole performance
misses the gate by a wide margin. Architecture A is therefore rejected. The
recommended Architecture B is two explicit kernels/contracts—`FastBounded`
when the caller proves the quotient bound and `Robust` otherwise—with no tensor
scan and no always-present masked fallback.

## SIMD dispatch risk

SFPI `v_if` is lane predication, not a free divergent branch. A long fallback
inside a lane mask still contributes its vector instructions to the issued
instruction stream. One unsafe lane may therefore cost almost as much as an
all-unsafe vector. No efficient vector-wide any-lane early exit has yet been
identified for this kernel. This is the primary architecture/performance
blocker for an always-present hybrid.

## Host exact-reference model

File:

```text
tools/sfpi_modulo_hybrid_reference.py
```

It uses exact `Fraction` arithmetic for FP32 values and NumPy binary32 candidate
operations. It implements the classifier and radix-2 magnitude fallback.

Last completed command:

```bash
cd /home/user/tt-metal
python3 tools/sfpi_modulo_hybrid_reference.py --random 1000000
```

Result:

```text
deterministic cases=110 unsafe=68 fallback_failures=0
BF16 exhaustive safe=151180 fallback=107168
random safe=581735 fallback=411380 seed=0x5f91
```

The BF16 run exhausts all positive finite BF16 bit patterns for eight selected
divisors. The random run covers one million seeded normal FP32 dividends. This
is not exhaustive over all FP32 pairs and does not model all SFPU special-value
details.

### Chunked exact-reference model

File:

```text
tools/sfpi_modulo_chunked_reference.py
```

The model evaluates requested `CHUNK_BITS` values `8`, `12`, `16`, `18`, and
`20`. For every non-excluded input it checks the one-sided quotient-estimate
bound, component product width, correction count, exact stage invariant, and
minimum exponent-gap progress. Each candidate completed:

```text
deterministic and boundary pairs       582
positive finite BF16 pairs          357632
seeded random normal FP32 pairs     1000000
total non-excluded pairs            1349689
failures                                  0
```

Candidate summary:

| Chunk bits | Fixed stages | Max observed | Components | Min progress | Failures |
|---:|---:|---:|---:|---:|---:|
| 8 | 37 | 18 | 2 x <=20 bits | 7 | 0 |
| 12 | 23 | 12 | 2 x <=16 bits | 11 | 0 |
| 16 | 17 | 9 | 2 x <=12 bits | 15 | 0 |
| 18 | 15 | 8 | 3 x <=10 bits | 17 | 0 |
| 20 | 14 | 7 | 3 x <=8 bits | 19 | 0 |

`CHUNK_BITS=16` minimizes fixed component subtractions among these candidates:
`74`, `46`, `34`, `45`, and `42`, respectively. Its full rerun reported 8,525
excluded pairs; all were classified as exact-intermediate/result subnormal and
therefore outside the current SFPU FTZ proof. Host verification is strong
finite-normal evidence, not exhaustive FP32 or device proof.

## Existing Blackhole validation and performance

Hardware detected on this server:

```text
Blackhole p150b, PCI device 5, board 000004123191f04e
```

Focused production caching-only tests already completed:

```text
14 passed, 6 expected harness skips
```

Blackhole MATH_ISOLATE cycles per tile:

| Variant | `fmod` | `remainder` | Status |
|---|---:|---:|---|
| Original baseline | 2976.242 | 3328.242 | incomplete old reducer |
| Caching only | 2912.227 | 3264.227 | preserves baseline behavior |
| Two corrections | 1568.227 | 1888.227 | rejected correctness candidate |
| Masked radix-2 hybrid | 91229.445 | 91901.445 | selected cases pass; performance architecture rejected |
| Chunked C16 robust lower bound | 37979.469 | 38683.461 | selected cases pass; Architecture A rejected |

Caching-only delta:

- `fmod`: 64.016 cycles/tile removed, about 2.151%, 1.0220x.
- `remainder`: 64.016 cycles/tile removed, about 1.923%, 1.0196x.

Caching-only Blackhole remainder disassembly:

- 65 encoded instructions versus 66 baseline;
- `sfpmul`: 1 versus 4 baseline;
- `ttreplay`: 5 in both;
- no observed spills.

The rejected two-correction speed numbers are only an upper bound and must
always be labelled unsafe.

The always-present radix-2 hybrid is 31.33x slower than caching-only fmod and
28.15x slower than caching-only remainder. Blackhole disassembly shows a
254-iteration scalar loop inside each of the 32 vector iterations. Each stage
issues 10 SFPU instructions plus two scalar loop instructions even when all
fallback lanes are predicated off.

The same experimental variants compile for Wormhole. Wormhole disassembly has
an additional `sfpnop` per fallback stage for post-MAD scheduling. This is
compile evidence only; no Wormhole hardware result exists.

No end-to-end TTNN measurement completed because the built `ttnn.device` module
was unavailable. No Wormhole hardware results exist.

## Documentation and artifact files

The following research artifacts are included in WIP commit `86ba7d8b5e`:

```text
docs/SFPI_SCALAR_MODULO_CORRECTNESS_AUDIT.md
docs/SFPI_SCALAR_MODULO_HYBRID_RESEARCH.md
docs/SFPI_SCALAR_MODULO_OPTIMIZATION.md
docs/disassembly/sfpi_remainder_before.disasm
docs/disassembly/sfpi_remainder_after.disasm
docs/source_snapshots/sfpi_remainder_before.txt
docs/source_snapshots/sfpi_remainder_after.txt
tools/sfpi_modulo_hybrid_reference.py
```

`docs/SFPI_SCALAR_MODULO_HYBRID_RESEARCH.md` now has 19 report sections and
labels claims as Proven, Host verified, Blackhole verified,
compile-only, Inferred, or Unknown. It now records the completed Blackhole
diagnostic, selected-case hybrid correctness, disassembly, and performance.

New chunked Architecture A report and host model:

```text
docs/SFPI_SCALAR_MODULO_CHUNKED_RESEARCH.md
tools/sfpi_modulo_chunked_reference.py
```

## Experimental Blackhole diagnostic/hybrid prototype

The following test-only files are in the WIP commit and have local follow-up
changes:

```text
tt_metal/tt-llk/tests/helpers/include/sfpu_scalar_modulo_hybrid.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_hybrid_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_hybrid.py
```

New local performance probes:

```text
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_hybrid_perf.cpp
tt_metal/tt-llk/tests/python_tests/perf_sfpu_scalar_modulo_hybrid.py
```

They do not modify the production SFPI API.

The helper contains:

- cached reciprocal/truncation fast candidate;
- the proven `exexp(scaled) < 22` normal-domain classifier;
- a 254-stage masked radix-2 fallback using `setexp` and `addexp`;
- signed `fmod` and floor-remainder prototypes for both scalar signs;
- a minimal branch-isolated raw-bit diagnostic.

Diagnostic layout:

```text
0      a
1      b
2      reciprocal
3      scaled
4      q_hat
5      residual before correction
6      residual after positive correction
7      residual after negative correction
8      safe predicate
9..31  repeated final corrected residual
```

Python tests cover:

- exact raw-bit intermediates for `a=50331656`, `b=3`;
- safe and unsafe values;
- positive and negative inputs;
- exact multiples;
- both scalar signs for fmod and floor remainder;
- the BF16-exact large counterexample as FP32 input.

Formatting/lint status for all five experimental files:

```text
pre-commit: all invoked hooks passed
git diff --check: passed
```

### Completed Blackhole diagnostic and correctness status

Command:

```bash
cd /home/user/tt-metal/tt_metal/tt-llk/tests/python_tests
CHIP_ARCH=blackhole PYTHONPATH=/tmp/tt_llk_test_deps \
python3 -m pytest test_sfpu_scalar_modulo_hybrid.py -q -x -vv
```

Result:

```text
5 passed
```

The minimal raw-bit diagnostic exactly confirmed:

```text
a                            0x4c400002
b                            0x40400000
reciprocal                   0x3eaaaaab
scaled                       0x4b800002
q_hat                        0x4b800002
residual before correction   0xc0800000  (-4)
after positive correction    0xc0800000  (-4)
after negative correction    0xbf800000  (-1)
classifier                   0x00000000  (unsafe)
```

The first checkpoint-heavy diagnostic caused the SFPI compiler ICE
`cannot store sfpu register (register spill)`. The minimal diagnostic computes
each output in a separate scalar branch and passes. The hybrid fmod and
floor-remainder tests pass bit-exactly for both scalar signs and the selected
positive/negative, safe/unsafe, exact-multiple, BF16-exact, and
`3*2^24`-neighbor cases. Floor remainder requires magnitude complementation
when operand signs differ and restoration of the dividend sign when the exact
result is zero.

### Completed performance and disassembly status

Command:

```bash
CHIP_ARCH=blackhole PYTHONPATH=/tmp/tt_llk_test_deps \
python3 -m pytest perf_sfpu_scalar_modulo_hybrid.py -q -x -vv
```

Result: `2 passed`. MATH_ISOLATE cycles/tile:

```text
fmod       91229.4453125
remainder  91901.4453125
```

Blackhole hybrid symbol sizes are 70 encoded instructions for fmod and 91 for
remainder; the dynamic stage loop is 10 SFPU plus two scalar instructions and
runs 254 times per vector iteration. The compiler uses `L0` through `L7` with
no spills. Wormhole compile-only symbol sizes are 79 and 100 instructions, with
one extra `sfpnop` per dynamic stage.

Wormhole compile-only command:

```bash
CHIP_ARCH=wormhole PYTHONPATH=/tmp/tt_llk_test_deps \
python3 -m pytest test_sfpu_scalar_modulo_hybrid.py \
    perf_sfpu_scalar_modulo_hybrid.py -q -x -vv --compile-producer
```

Result: `7 passed` in compile-producer mode across the correctness and
performance sources. This is not runtime validation.

## Fixed-stage chunked Architecture A prototype

Test-only files:

```text
tt_metal/tt-llk/tests/helpers/include/scalar_modulo_chunked_research.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_chunked_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_chunked.py
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_chunked_perf.cpp
tt_metal/tt-llk/tests/python_tests/perf_sfpu_scalar_modulo_chunked.py
```

The direct `q*d` formulation is not proof-safe: SFPMAD preserves four product
bits beyond binary32, so the effective product width is 28 significant bits.
A 16-bit quotient times a 24-bit divisor can lose remainder-significant low
bits. The prototype instead splits the divisor into two <=12-bit components and
uses one-sided quotient estimation. A zero low component must not be passed
directly through `setexp`, because raw exponent replacement would manufacture a
power of two; its scaling is predicated and the corresponding product remains
zero.

The straightforward combined cached-fast-path plus chunked fallback exceeded
the SFPU register budget and failed compilation with `cannot store sfpu
register`. A standalone in-place robust reducer compiles without a spill and is
an honest lower bound on the cost of the combined hybrid. The functional and
performance sources intentionally measure that standalone reducer.

Final Blackhole compile and runtime commands:

```bash
CHIP_ARCH=blackhole PYTHONPATH=/tmp/tt_llk_test_deps \
python3 -m pytest test_sfpu_scalar_modulo_chunked.py \
    perf_sfpu_scalar_modulo_chunked.py -q -x -vv --compile-producer

CHIP_ARCH=blackhole PYTHONPATH=/tmp/tt_llk_test_deps \
python3 -m pytest test_sfpu_scalar_modulo_chunked.py -q -x -vv

CHIP_ARCH=blackhole PYTHONPATH=/tmp/tt_llk_test_deps \
python3 -m pytest perf_sfpu_scalar_modulo_chunked.py -q -x -vv
```

Results:

```text
Blackhole final compile-only       36 passed
Blackhole functional/raw runtime   28 passed
Blackhole performance runtime       8 passed
```

The raw-bit diagnostic passes for divisors `3`, `5`, `7`, and `10` on the
original disputed large inputs. Signed fmod and floor-remainder tests pass for
both scalar signs with `3`, `5`, `7`, `10`, FP32 `0.1`, and FP32 `0.3`. This is
selected normal-domain validation, not exhaustive device correctness.

Blackhole MATH_ISOLATE performance is identical for 0, 1, 16, and 32 unsafe
lanes because the robust stage loop is always present in the issued instruction
stream:

| Operation | Cycles/tile | Text bytes | Versus caching-only |
|---|---:|---:|---:|
| fmod | 37979.46875 | 2635 | 13.04x slower |
| remainder | 38683.4609375 | 2723 | 11.85x slower |

This is about 58% cheaper than the rejected masked radix-2 prototype but still
far above the 10,000-cycle rejection threshold. The combined hybrid cannot be
cheaper than this standalone lower bound and does not compile without a spill.
Architecture A is rejected.

Final Blackhole disassembly:

- fmod reducer: 344 bytes, 86 encoded instructions;
- remainder reducer: 432 bytes, 108 encoded instructions;
- dynamic stage: 66 SFPU instructions plus two scalar loop instructions,
  repeated 17 times within each 32-vector tile loop;
- visible local use is `L0` through `L6`; no compiler spill and no replay.

Final Wormhole compile-only command uses the same two sources and
`CHIP_ARCH=wormhole`; result: `36 passed`. Wormhole symbol sizes are 372 bytes
(93 instructions) for fmod and 460 bytes (115 instructions) for remainder. Its
stage has 73 SFPU instructions plus two scalar loop instructions, including five
stage-local `sfpnop` instructions. No Wormhole runtime hardware was available.

Final hygiene checks:

```text
pre-commit on all modified/new research files: all invoked hooks passed
git diff --check: passed
production-header diff/status: empty
```

## Architecture B robust fixed-schedule result

The test-only scalar-specialized reducer precomputes a fixed sequence:

```text
K0=max(112-Eb,0)
K=K0,K0-15,...,0
d=b*2^K
```

Each local ratio is below `2^16`, so the general FP32 truncation sequence is
replaced by portable `FP32 -> UINT16 -> FP32` conversion. The two-instruction
conversion plus a two-component 16b-by-12b SFPMAD subtraction keeps the local
quotient error in `{0,+1}` and needs one negative correction.

New test-only files:

```text
tools/sfpi_modulo_fixed_schedule_reference.py
tools/sfpi_modulo_fixed_schedule_exhaustive.cpp
tt_metal/tt-llk/tests/helpers/include/scalar_modulo_fixed_schedule_research.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_fixed_schedule_test.cpp
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_fixed_schedule_perf.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_fixed_schedule.py
tt_metal/tt-llk/tests/python_tests/perf_sfpu_scalar_modulo_fixed_schedule.py
docs/SFPI_SCALAR_MODULO_FIXED_SCHEDULE_RESEARCH.md
```

Host results:

```text
selected deterministic/BF16/random:
  tested=456700 failures=0 exclusions=1514 q_error=[0,1]
  schedule_stages=17 active_stages=9 max_q=32768 prehalved=1842

arbitrary normal divisor/random:
  tested=95002 failures=0 exclusions=5580 q_error=[0,1]
  schedule_stages=17 active_stages=17 max_q=32768 prehalved=366
```

Exclusions remain explicit: subnormal/FTZ results and components, reciprocal
range, and component-subtraction subnormal transients. The exact exponent-127
pre-reduction closed the three observed top-of-range partial-product overflows.

The optimized C++ verifier exhaustively processed all `2,130,706,432` positive
normal FP32 dividends for each of `3`, `5`, `7`, `10`, FP32 `0.1`, FP32 `0.3`,
`8`, `nextDown(8)`, and `nextUp(8)`: all `19,176,357,888` cases passed with no
exclusion or mismatch. With the smallest normal divisor it processed
`21,307,064,320` total cases with zero mismatch, but explicitly classified
184,549,377 smallest-divisor cases as `IntermediateSubnormal`. The largest
normal divisor is outside the configuration domain because its reciprocal is
subnormal.

Final device/compiler results:

```text
Blackhole compile-only                  40 passed
Wormhole compile-only                   40 passed
Blackhole functional/diagnostic runtime 32 passed
Blackhole perf runtime                   8 passed
```

The raw-bit matrix covers both operations, both scalar signs, divisors `3`,
`5`, `7`, `10`, FP32 `0.1`, FP32 `0.3`, and power-of-two `8`. Each case now
contains 231 seeded normal values plus eight exponent-127 cases in addition to
boundaries/counterexamples. Expected bits now come from an exact integer-lattice
oracle instead of PyTorch.

Blackhole p150b MATH_ISOLATE for scalar `3`:

| Operation | Cycles/tile | Text bytes | Versus C16 | Versus caching-only |
|---|---:|---:|---:|---:|
| fmod | 6171.445 | 2610 | 6.154x faster | 2.119x slower |
| remainder | 6875.445 | 2698 | 5.626x faster | 2.106x slower |

The 0/1/16/32 active-lane results are effectively identical. Blackhole fmod is
312 bytes/78 instructions/70 SFPU instructions; remainder is 400 bytes/100/92.
Both symbols have no spill loads/stores. Wormhole symbols compile at 392 and
480 bytes with no spills; Wormhole runtime remains unknown.

Blackhole tests with the smallest normal divisor confirmed the remaining FTZ
boundary. Positive subnormal subtraction results flush to `+0`; a negative
subnormal can flush to `-0` and enter the negative correction, producing an
extra divisor. This agrees with the documented SFPMAD denormal flush behavior.

Decision: Architecture B is the preferred direction. Keep `FastBounded` as a
separate explicit caller-contract specialization and continue hardening this
fixed schedule as `Robust`. Do not call it globally robust until the documented
finite-normal exclusions are closed.

## Exponent-stationary host result

The host-only follow-up keeps the reduction divisor at unbiased exponent 111,
scales the residual upward between stages, and forms the physical normal or
subnormal result with an exact integer RNE pack. Mandatory exponent-127
pre-halving permits the corrected initial schedule `K0=max(111-Eb,0)`; using the
old `112-Eb` schedule in a 111 frame would underflow the smallest inputs during
initial scaling.

The primary smallest-normal-divisor gate passed:

```text
Passed                 2,130,706,432
IntermediateSubnormal              0
other exclusions                   0
failures                           0
final subnormal results   184,549,377  # packed as raw bits
```

The old physical-frame control histogram locates every one of its 184,549,377
exclusions at `AfterHighSubtract`; all have subnormal exact final results and
96,467,969 are negative before correction.

Complete positive-normal input sweeps passed for 14 divisors: four exponent
`-126` mantissa boundaries, the previous nine representative divisors, and
`FLT_MAX`. Total: `29,829,890,048` accepted pairs with no exclusion or mismatch.
The independent Python model passed 458,214 deterministic/BF16/selected cases
and 100,582 arbitrary-normal-divisor cases with zero exclusion and quotient
error `[0,+1]`. See
`docs/SFPI_SCALAR_MODULO_EXPONENT_STATIONARY_RESEARCH.md`.

This remains host evidence for the reducer. The separate device transport gate
below does not validate stationary arithmetic or its register allocation.

## Raw-pack transport device gate

Test-only files:

```text
tt_metal/tt-llk/tests/sources/sfpu_raw_pack_transport_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_raw_pack_transport.py
docs/SFPI_SCALAR_MODULO_RAW_PACK_TRANSPORT.md
```

The microkernel constructs FP32 bit patterns directly in SFPU LRegs, stores
them to 32-bit Dst, packs to a UInt32 output buffer, and compares host uint32
words. It includes signed zeros, 13 positive/negative subnormal patterns,
min-normal boundaries, ordinary normals, and maximum finite values.

Final compile and silicon runs passed on both architectures:

```text
Wormhole compile/silicon  2/2 passed
Blackhole compile/silicon 2/2 passed
```

The passing contract is intentionally architecture-specific:

- Blackhole `MOD0_FMT_INT32` (`4`) plus UInt32 pack preserves every tested word
  exactly, including all subnormal encodings.
- Wormhole `MOD0_FMT_INT32` flushes subnormal-looking encodings. A stronger
  attempt using pre-rotated numeric `MOD0=9` opaque-32 store plus UInt32 pack
  still flushes every subnormal encoding to the corresponding signed zero.
- FP32 control stores flush only the subnormal encodings on both architectures.
- Disabling packer zero-flag substitution did not change Wormhole output, so
  that is not the observed conversion point.

Disassembly confirms the final raw forms:

```text
Blackhole  sfpstore ...,0,4,7
Wormhole   sfpstore ...,0,9,3
```

Wormhole also requires point-of-use `sfpconfig; sfpnop; sfpencc` after the
32-bit unpack-to-Dst handshake; prelude-only initialization left stores
ineffective. No production header was changed.

Decision: the exact final-subnormal output gate is open for Blackhole and
closed for Wormhole through all tested Dst/packer paths. A common exact robust
kernel is therefore blocked. Wormhole reducer research may continue only with
an explicit FTZ result contract or after a separately verified output mechanism
is found.

## Blackhole stationary normalized-reducer device gate

Test-only files:

```text
tt_metal/tt-llk/tests/helpers/include/scalar_modulo_stationary_research.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_stationary_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_stationary.py
docs/SFPI_SCALAR_MODULO_STATIONARY_DEVICE_RESEARCH.md
```

The isolated kernel implements only:

```text
abs(input)
    -> exponent-127 pre-half
    -> stationary D/rho/high/low stages at working exponent 111
    -> (2R) mod D top reconstruction
    -> normalized nonnegative FP32 R
```

There is no sign restoration or physical/raw finalizer. The exact raw-bit gate
passes all 11 requested divisor families on Blackhole p150b:

```text
Blackhole compile  11/11 passed
Blackhole silicon  11/11 passed
```

The matrix covers four `Eb=-126` mantissa/split boundaries, 3, FP32 0.1,
`nextDown(8)`, 8, `nextUp(8)`, `2^112`, and `FLT_MAX`. It includes `a<b`,
equality, exact multiples, one-lattice-step neighbors, local quotients 65534
and 65535, ratios below 65536, both signs, and exponent-127 pre-half
boundaries wherever representable. There are 11,264 device result comparisons
from 299 designed signed patterns. The independent stage oracle checks 1,000
active stages in the designed set, observes quotient errors `{0,+1}`, and
rejects any `q_hat` outside `{q,q+1}`.

All reducer symbols compile without a spill. One-stage forms are `0xa8` bytes;
multi-stage forms are `0xf8` bytes. Only `L0-L4` are visible. Each symbol has
the two expected input/reconstruction `sfpload` instructions, one output
`sfpstore`, and no scalar-memory operations. The two `ttreplay` instructions in
multi-stage forms replay the stationary stage; they are not spills.

This closes the selected Blackhole normalized-arithmetic and register gate. It
does not validate sign semantics, physical normal/subnormal packing,
special/zero-divisor policy, stationary performance, Wormhole arithmetic, or a
production operator.

## Exact next steps

1. Keep Architecture A rejected; do not combine the cached fast path with an
   always-issued fallback.
2. Keep the passing normalized exponent-stationary magnitude reducer isolated
   from finalization; use it as the Blackhole arithmetic/register baseline.
3. For Blackhole only, add the integer normal/subnormal finalizer, perform
   floor-remainder `D-R` sign adjustment before packing, preserve exact-zero
   sign, and add the high-divisor `a<b` bypass.
4. For Wormhole, either approve an explicit FTZ result contract or first build
   and gate a genuinely different transport such as split 16-bit output. Do not
   present the current raw/UInt32 path as exact.
5. Compile/disassemble both reducers, reject spills, and run raw-bit,
   top-range, correctness, register-pressure, and lane-mix performance tests.
6. Define special-value, zero-divisor, signed-zero, subnormal-input, and FTZ
   policy at the operator boundary.
7. Review the two explicit Architecture B contracts with the TT/SFPI owner:
   `FastBounded` selected only from proven range metadata, and fixed-schedule
   `Robust` as the generic path after its exclusions are closed.
8. Obtain Wormhole reducer runtime correctness/performance and tune its
   dependency schedule; current Wormhole runtime evidence covers transport only.
9. Make no production change until the robust proof domain, per-architecture
   output semantics, and API/dispatch contract are approved; then rerun host,
   raw-bit device, disassembly, register, lane-mix, and TTNN validation.

## Safety and claim rules

- Do not push or commit unless the user explicitly asks again.
- Do not stage `tt-isa-documentation/`.
- Preserve unrelated working-tree changes.
- Do not call the old ten-block reducer mathematically correct.
- Do not call the two-correction commit correct.
- Blackhole chunked verification covers selected normal cases for divisors
  `3`, `5`, `7`, `10`, FP32 `0.1`, and FP32 `0.3`; do not generalize it to the
  full FP32/FTZ/special-value domain.
- Do not claim Wormhole exact subnormal output: real Wormhole hardware flushed
  every tested subnormal encoding through both INT32 and opaque-32/UInt32 paths.
- Keep special-value policy explicitly unresolved. Treat FTZ/subnormal behavior
  as measured only for the documented smallest-normal-divisor Blackhole probes;
  do not generalize it to all inputs or to Wormhole without hardware evidence.
- Treat the exponent-stationary all-normal-divisor proof as host evidence over
  its documented sweep. Blackhole device evidence covers only the selected
  normalized-reducer matrix above; performance and physical finalization are
  still unvalidated.
