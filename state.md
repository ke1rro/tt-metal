# SFPI Scalar Modulo Research — Work State

Last updated: 2026-07-18 UTC

This file is a restart/handoff record for the scalar SFPI `fmod` and floor-style
`remainder` optimization research. It intentionally records unfinished work as
well as verified results.

## Repository and Git state

- Checkout: `/home/user/tt-metal`
- Branch: `opt/optimize-sfpi-scalar-modulo`
- HEAD: `375862277bcb2b3ef89a2543b090f8dbf3eb49d7`
- HEAD subject: `Optimize scalar SFPI fmod and remainder`
- Tracking remote: `fork/opt/optimize-sfpi-scalar-modulo`
- Fork: `git@github.com:ke1rro/tt-metal.git`
- Git identity requested by the user:
  - username: `ke1rro`
  - email: `nikitalenyk2@gmail.com`
- No new commit or push was made after the correctness audit.
- The remote branch and HEAD still contain the rejected two-correction
  implementation. Do not present commit `375862277bc` as correctness-safe.
- The local working tree changes production headers back to the old ten-block
  correction behavior while retaining only cached `scaled = v * recip_val`.
  This local rollback is uncommitted and unpushed.
- `tt-isa-documentation/` is an untracked user-supplied directory. Do not stage,
  edit, or delete it.

## User goal

Research, mathematically justify, prototype, validate, disassemble, and benchmark
a correctness-preserving hybrid SFPI implementation:

```text
proven safe quotient range
    -> cached reciprocal product + truncation + two corrections

unsafe/large quotient range
    -> robust exponent-scaled range reduction
```

The full design brief is in:

```text
/home/user/.codex/attachments/85cf7851-d51f-4d10-b3b0-adf43a363e57/pasted-text.txt
```

Do not integrate the hybrid into production until classifier and fallback have
passed exact host tests, Blackhole diagnostics, prototype correctness, and a
performance assessment. Wormhole results must remain explicitly unknown on
this Blackhole-only server.

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

### Fallback conclusion

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

Radix-4 remains the next performance candidate. Radix-16 is expected to have
high register/predicate cost and has not been validated.

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

No end-to-end TTNN measurement completed because the built `ttnn.device` module
was unavailable. No Wormhole hardware results exist.

## Documentation and artifact files

Created locally and currently uncommitted:

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

`docs/SFPI_SCALAR_MODULO_HYBRID_RESEARCH.md` has all 18 requested report
sections and labels claims as Proven, Host verified, Blackhole verified,
Wormhole verified, Inferred, or Unknown. It currently states Blackhole raw-bit
diagnostics and hybrid device performance as unknown; update it only after the
experimental tests below pass.

## Experimental Blackhole diagnostic/hybrid prototype

The following test-only files were added and are uncommitted:

```text
tt_metal/tt-llk/tests/helpers/include/sfpu_scalar_modulo_hybrid.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_hybrid_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_hybrid.py
```

They do not modify the production SFPI API.

The helper contains:

- cached reciprocal/truncation fast candidate;
- the proven `exexp(scaled) < 22` normal-domain classifier;
- a 254-stage masked radix-2 fallback using `setexp` and `addexp`;
- signed positive-divisor `fmod` and floor-remainder prototypes;
- a 32-vector diagnostic output layout.

Diagnostic layout:

```text
0..8   a, b, reciprocal, scaled, q_hat, pre-correction residual,
       safe predicate, fast residual, initial scaled divisor
9..28  fallback residual after steps 0..19
29     fallback residual after step 23
30     final fallback magnitude
31     final positive-divisor fmod result
```

Python tests cover:

- exact raw-bit intermediates for `a=50331656`, `b=3`;
- safe and unsafe values;
- positive and negative inputs;
- exact multiples;
- both fmod and positive-divisor floor remainder;
- the BF16-exact large counterexample as FP32 input.

Formatting/lint status for the three experimental files:

```text
pre-commit: all invoked hooks passed
git diff --check: passed before the last two-line compile fix
```

### Last device-run status

Command:

```bash
cd /home/user/tt-metal/tt_metal/tt-llk/tests/python_tests
python3 -m pytest test_sfpu_scalar_modulo_hybrid.py -q -x -vv
```

Attempt 1 failed before SFPI compile because runtime `formats` was not bound.
That was fixed by adding the same `params.formats` guards used by the standard
unary test source.

Attempt 2 reached the math compile and reported two C++ issues:

```text
signed/unsigned comparison: block < params.NUM_BLOCKS
non-dependent static_assert in a discarded if constexpr branch
```

Both were just fixed locally:

- math and pack block loop counters changed from `std::uint32_t` to `int`;
- the non-dependent static assertion was removed.

The server interruption happened immediately after applying that fix. The test
has not yet been rerun, so no Blackhole raw-bit or hybrid result from this new
prototype may be claimed yet.

## Exact next steps after restart

1. Confirm files and formatting:

   ```bash
   cd /home/user/tt-metal
   git status -sb
   git diff --check
   pre-commit run --files \
     tt_metal/tt-llk/tests/helpers/include/sfpu_scalar_modulo_hybrid.h \
     tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_hybrid_test.cpp \
     tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_hybrid.py
   ```

2. Rerun the experimental device tests:

   ```bash
   cd /home/user/tt-metal/tt_metal/tt-llk/tests/python_tests
   python3 -m pytest test_sfpu_scalar_modulo_hybrid.py -q -x -vv
   ```

3. If the SFPI compiler rejects nested predicates, dynamic `setexp`, or the
   checkpoint selector, change only the experimental helper/source. Do not edit
   the four production headers for the prototype.

4. Once the diagnostic passes, capture and record exact Blackhole bits for:

   ```text
   a          0x4c400002
   b          0x40400000
   reciprocal 0x3eaaaaab
   scaled     0x4b800002
   q_hat      0x4b800002
   residual   expected -4.0
   classifier expected 0.0 (unsafe)
   fast result expected -1.0 (invalid)
   fallback   expected 2.0
   ```

5. Validate mixed-lane behavior and exact signed results. Expand tests for
   negative scalar semantics before any production recommendation.

6. Add a test-only MATH_ISOLATE perf source or mode and measure:

   - all safe lanes;
   - one unsafe lane;
   - half unsafe lanes;
   - all unsafe lanes.

   The expected issue is that masked fallback instruction cost remains even for
   safe lanes.

7. Generate experimental math disassembly and record static instruction count,
   `sfpmul`, MAD, comparisons, replay blocks, register allocation, and spills.

8. Update `docs/SFPI_SCALAR_MODULO_HYBRID_RESEARCH.md` sections 10, 12, 13,
   14, 15, and 18 with measured Blackhole results.

9. Decide between:

   - an always-present per-lane hybrid;
   - a host-visible safe/robust kernel variant;
   - a radix-4 fallback prototype.

10. Only after the above, make the smallest production patch and rerun focused
    correctness, LLK performance, disassembly, and any available TTNN tests.

## Safety and claim rules

- Do not push or commit unless the user explicitly asks again.
- Do not stage `tt-isa-documentation/`.
- Preserve unrelated working-tree changes.
- Do not call the old ten-block reducer mathematically correct.
- Do not call the two-correction commit correct.
- Do not call host fallback results Blackhole verified.
- Do not claim Wormhole equivalence without Wormhole hardware or a saved
  Wormhole reference.
- Keep FTZ/subnormal and special-value behavior explicitly unknown until tested
  against the documented SFPU contract.
