# SFPI scalar modulo combined stationary Blackhole gate

## Result

The test-only combined stationary kernel passes its selected end-to-end
correctness, Blackhole lowering, register, and performance gates on p150b:

```text
physical normal input magnitude
    -> stationary normalized reduction
    -> exponent-127 top reconstruction
    -> exact-zero decision
    -> optional normalized D-R
    -> integer normal/subnormal RNE pack
    -> sign restoration
    -> INT32 SFPSTORE
    -> UInt32 pack
```

Final results:

```text
Correctness specializations compiled   64/64
Correctness specializations on silicon 64/64
Raw UInt32 comparisons                 65,536
Performance specializations compiled   42/42
Performance specializations on silicon 42/42
Stationary quotient error              {0,+1} in the independent host oracle
Explicit SFPU locals                    L0-L7 maximum
Scalar-memory operations                0
Input sfpload / final raw sfpstore       1 / 1 per combined symbol
```

This joins the two previously isolated Blackhole phases. The finalizer baseline
was first committed separately as `65ea79474b` (`research: validate Blackhole
stationary modulo finalizer`), so the combined experiment does not erase the
last independently proven boundary.

This is not a production integration and does not establish behavior for all
FP32 pairs, special values, zero divisors, subnormal inputs, or Wormhole.

## Test-only files

```text
tt_metal/tt-llk/tests/helpers/include/scalar_modulo_stationary_combined_research.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_stationary_combined_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_stationary_combined.py
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_stationary_combined_perf.cpp
tt_metal/tt-llk/tests/python_tests/perf_sfpu_scalar_modulo_stationary_combined.py
```

The helper reuses the committed stationary reducer and integer finalizer. No
production SFPI header, LLK enum, or operator dispatch is changed.

## Combined phase boundary

The combined function keeps the reducer result as the only value crossing the
reduction/finalization boundary:

```cpp
sfpi::vFloat normalized_r;
{
    normalized_r = reduce_scalar_modulo_stationary_combined<...>(input_magnitude);
}
{
    output_bits = finalize_stationary_magnitude<...>(normalized_r);
}
```

The scopes state the intended lifetime but are not used as evidence by
themselves. The evidence is the final disassembly: the compiler reuses the
eight explicit SFPU locals, emits no scalar spill memory, and does not reload an
intermediate result from Dst.

The executed order is:

```text
1. Load and take the input magnitude.
2. Apply the mandatory exponent-127 pre-half and initial normalization.
3. Run the stationary D/rho/high/low stages.
4. Reconstruct (2R) mod D after a top-range pre-half.
5. Test exact R == 0 before any floor adjustment.
6. For differing-sign floor remainder and nonzero R, form D-R.
7. Assemble physical normal/subnormal FP32 bits with integer RNE.
8. Restore the required sign.
9. Store once with Blackhole INT32 raw transport and pack as UInt32.
```

The zero decision precedes `D-R`, so an exact multiple remains signed zero
rather than becoming the divisor. Fmod zero and nonzero values use the dividend
sign. Floor-remainder nonzero values use the divisor sign, while exact zero
uses the dividend sign.

## High-divisor bypass

For `Eb>111`, scaling a physical `a<b` input down into the exponent-111 frame
can underflow before the stationary reducer begins. The combined kernel tests
`a<D` in the physical frame and puts bypass and reduction into disjoint
predicated paths:

```text
a < D:
    fmod or same-sign floor magnitude = a
    differing-sign nonzero floor magnitude = D-a

a >= D:
    stationary reducer -> integer finalizer
```

Both branches produce positive raw magnitude bits. Sign restoration happens
once after they merge. This code shape was necessary for compiler liveness: an
earlier form that carried bypass state across the reducer produced a Blackhole
compiler internal error in the high-exponent floor specialization.

The physical `D-a` bypass does not enter the subnormal range for the selected
normal `Eb=112` and `FLT_MAX` boundaries. Subnormal inputs remain outside this
gate.

## Exact host oracle and matrix

The expected final word is computed directly from the integer representation
of each binary32 operand:

```text
exact integer-lattice remainder
    -> optional exact divisor-minus-remainder
    -> integer round-to-nearest-even FP32 composition
    -> exact sign bit
```

NumPy is used to construct binary32 constants and reciprocal candidates, but
NumPy/PyTorch `fmod` or remainder is not the expected-result oracle.

The 64 compile-time specializations are eight divisors times two operations
times four operand-sign combinations:

| Divisor | `Eb` | Primary boundary |
|---:|---:|---|
| `0x40400000` (`3.0`) | 1 | ordinary normal output and multi-stage reduction |
| `0x00800000` | -126 | 17-stage reduction and full subnormal pack |
| `0x00800001` | -126 | nontrivial lattice and split boundaries |
| `0x0c000000` | -103 | normal-only physical-result boundary |
| `0x0b800000` | -104 | first subnormal-capable result class |
| `0x77000000` | 111 | one-stage stationary specialization |
| `0x77800000` | 112 | negative initial scale and physical bypass |
| `0x7f7fffff` | 127 | top range, pre-half, and physical bypass |

Each divisor set selects all representable members of these classes:

```text
zero and minimum normal input
a < b
a == b
exact multiple
next representable value above the multiple
floor(a/b) = 65534
floor(a/b) = 65535
ratio immediately below 65536
exponent-127 and FLT_MAX boundaries
positive/negative dividend and divisor
fmod and floor remainder
normal, subnormal, and exact-zero results where representable
```

There are 123 distinct divisor/magnitude patterns and 984
operation/sign-expanded designed patterns before tile repetition. The 64 full
32x32 tiles yield 65,536 raw result comparisons.

The independent stationary host model checks 215 active quotient estimates in
the distinct designed magnitude sets. It observes 137 exact estimates and 78
one-high estimates, and asserts at every active stage:

```text
q_hat - q in {0,+1}
```

The device returns final raw bits, not a separate quotient trace. Thus the
quotient bound is host-model evidence paired with silicon final-result evidence,
not a claim that `q_hat` was independently transported from the device.

## Correctness and silicon result

The final code passed:

```text
Blackhole compile  64/64
Blackhole silicon  64/64
```

The host compares UInt32 word multisets because SFPU lane-to-tile placement is
outside this arithmetic/transport gate. The test still detects every missing,
extra, or mis-encoded word and reports count deltas by raw bit pattern.

The silicon debug history exposed two integration-specific failures that the
isolated phases could not reveal:

1. Keeping high-divisor bypass bits live through the reducer exceeded the
   acceptable compiler code shape; disjoint predicated branches fixed it.
2. Adding the floor result sign before testing zero changed positive-dividend,
   negative-divisor exact multiples into `-0`. Testing the positive raw
   magnitude before unified sign restoration fixed the required signed zero.

## Disassembly and register gate

All 64 final MATH ELFs contain the combined symbol. The automated symbol-body
audit reports:

```text
64 symbols found
64 with exactly one sfpload
64 with exactly one sfpstore ...,0,4,7
64 with zero lw/sw/lb/lh/lbu/lhu/sb/sh/flw/fsw
64 with no unexpected Dst reload
maximum explicit locals L0-L7
```

`L9`, `L11`, `L12`, `L13`, and, for the physical-bypass forms, `L14` are
architectural/programmed constants rather than allocated explicit locals. No
symbol uses `L8` or `L10` as a local. The size bounds are:

| Code shape | Bytes |
|---|---:|
| multi-stage fmod | `0x1d4-0x1dc` |
| multi-stage floor remainder | `0x1d4-0x204` |
| `Eb=111` fmod | `0xc4-0xcc` |
| `Eb=111` floor remainder | `0xc4-0xf4` |
| `Eb>111` fmod with bypass | `0x104-0x10c` |
| `Eb>111` floor with bypass | `0x104-0x144` |

Forty multi-stage symbols contain two `ttreplay` instructions for the
stationary stage body. These are instruction replays, not memory spills.

The combined `Eb=-104` symbol still lowers the finalizer classifier as the
passing less-than-first form. Its relevant sequence includes:

```text
sfpexexp ...
sfpiadd ...,0xFA7,1
```

The `MOD1=1` condition is the LT-first selector. The previously rejected
GTE-first shape did not reappear after inlining the reducer and finalizer.

## Performance method

The performance test compiles three phases separately under `MATH_ISOLATE`:

```text
phase 0: normalized reducer only
phase 1: isolated finalizer only
phase 2: combined reducer + finalizer
```

It uses eight tiles, loop factor 16, four faces, FP32 Dst accumulation, and
reports post-processed `TILE_LOOP` cycles per tile. Fmod uses same positive
signs. Floor remainder uses positive dividend/negative divisor so the measured
finalizer includes the nonzero `D-R` path. Seven divisor classes give 42
specializations, all of which compiled and ran on Blackhole silicon.

### Fmod cycles per tile

| Divisor | Reducer | Finalizer | Combined | Combined - reducer | Difference from finalizer |
|---|---:|---:|---:|---:|---:|
| `3.0` | 6203.430 | 1915.469 | 7963.688 | 1760.258 | -155.211 |
| smallest normal | 11067.422 | 1915.516 | 12827.672 | 1760.250 | -155.266 |
| `Eb=-103` | 10459.422 | 1915.516 | 12219.672 | 1760.250 | -155.266 |
| `Eb=-104` | 10459.422 | 1915.516 | 12219.672 | 1760.250 | -155.266 |
| `Eb=111` | 1307.453 | 475.422 | 1627.422 | 319.969 | -155.453 |
| `Eb=112` | 1403.422 | 475.469 | 2171.422 | 768.000 | +292.531 |
| `FLT_MAX` | 1403.430 | 475.445 | 2171.430 | 768.000 | +292.555 |

### Floor-remainder cycles per tile

| Divisor | Reducer | Finalizer | Combined | Combined - reducer | Difference from finalizer |
|---|---:|---:|---:|---:|---:|
| `3.0` | 6203.430 | 2267.477 | 8347.461 | 2144.031 | -123.445 |
| smallest normal | 11067.422 | 2267.469 | 13211.438 | 2144.016 | -123.453 |
| `Eb=-103` | 10459.422 | 2267.461 | 12603.438 | 2144.016 | -123.445 |
| `Eb=-104` | 10459.422 | 2267.469 | 12603.438 | 2144.016 | -123.453 |
| `Eb=111` | 1307.453 | 827.430 | 2011.422 | 703.969 | -123.461 |
| `Eb=112` | 1403.422 | 827.430 | 2683.430 | 1280.008 | +452.578 |
| `FLT_MAX` | 1403.430 | 827.445 | 2683.469 | 1280.039 | +452.594 |

For `Eb<=111`, combined overhead is consistently slightly below the isolated
finalizer: about 155 cycles/tile lower for fmod and 123 lower for floor
remainder. This is consistent with eliminating duplicated phase setup/load work
in the combined symbol.

For `Eb>111`, the required physical-frame `a<b` bypass changes the code shape.
Combined overhead is 768 cycles/tile for fmod and about 1280 for floor
remainder, respectively about 293 and 453 cycles above the isolated
normal-only finalizer. This is real integration cost, not a spill or reload.

The smallest-normal general form remains above 10k cycles/tile. The performance
result therefore validates the integration accounting but does not by itself
make this a production-speed modulo replacement.

## Commands

```bash
bash tt_metal/tt-llk/.claude/scripts/run_test.sh compile \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole \
  --test test_sfpu_scalar_modulo_stationary_combined.py --verbose

bash tt_metal/tt-llk/.claude/scripts/run_test.sh simulate \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole \
  --test test_sfpu_scalar_modulo_stationary_combined.py --verbose

bash tt_metal/tt-llk/.claude/scripts/run_test.sh compile \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole \
  --test perf_sfpu_scalar_modulo_stationary_combined.py --verbose

bash tt_metal/tt-llk/.claude/scripts/run_test.sh simulate \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole \
  --test perf_sfpu_scalar_modulo_stationary_combined.py --verbose
```

The runner's `simulate` phase consumes the compiled ELF on locked Blackhole
hardware. The performance CSV is written under
`tt_metal/tt-llk/perf_data/perf_sfpu_scalar_modulo_stationary_combined/` by the
standard LLK perf-report pipeline.

## Decision and remaining scope

The selected Blackhole combined gate is closed: the stationary reducer and
exact integer finalizer coexist in one symbol without spill, pass end-to-end
raw-bit comparisons, and have separately measured integration cost.

The next useful test-only optimization is the compile-time finalizer split:

```text
Eb >= -103: normal-only finalizer without subnormal integer RNE
Eb <  -103: full exact normal/subnormal finalizer
```

That split should retain the general combined kernel as the correctness
baseline and repeat the raw-bit, disassembly, and performance gates before any
production change.

Still unresolved:

1. NaN, infinity, zero divisor, subnormal input, and operator-level FTZ policy.
2. Exhaustive coverage of arbitrary FP32 dividend/divisor pairs.
3. Production API/dispatch ownership and comparison with the existing operator.
4. Wormhole exact final transport; the tested path still flushes subnormal
   encodings and cannot inherit this Blackhole claim.
5. End-to-end TTNN/operator validation after an integration is approved.
