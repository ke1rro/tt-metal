# SFPI scalar modulo exponent-stationary host research

## 1. Result

The host-only `Exponent-Stationary Fixed Schedule` passed its primary gate for
the smallest normal FP32 divisor:

```text
divisor_bits           0x00800000
positive normal inputs 2,130,706,432
Passed                 2,130,706,432
IntermediateSubnormal  0
other exclusions       0
failures               0
```

Of those inputs, 184,549,377 have an exact subnormal physical result. They are
still counted as passes because reduction remains normal and an integer
round-to-nearest-even pack creates the final raw FP32 bits.

Thirteen additional complete positive-normal sweeps also passed, for a total
of `29,829,890,048` accepted input/divisor cases with no exclusion or mismatch.
This validates a host architecture and justifies an SFPI prototype; it is not
yet device or production evidence.

## 2. Schedule correction enabled by top pre-halving

Let the positive normal divisor be:

```text
b = m_b * 2^Eb
```

and choose working exponent `C=111`. The normalized constant divisor is:

```text
D = b * 2^(C-Eb)
```

so `D` always has unbiased exponent 111. The first physical shift is:

```text
K0 = max(C-Eb, 0)
```

This is one smaller than the physical-frame schedule's `max(112-Eb,0)`. The
difference is intentional: exponent-127 dividends are first halved exactly, so
the largest working dividend has exponent 126. Consequently its ratio to a
divisor with exponent 111 is strictly below `2^16`.

Using the old `112-Eb` start with `C=111` would scale every initial residual by
one half and would itself underflow the smallest normal inputs. Starting at
`111-Eb` avoids that defect while preserving the UINT16 local quotient bound.

For `Eb<=C`, the initial normalized residual is the dividend (or its exact half
for exponent 127). After a stage, `R<D`; before the next stage:

```text
R = R * 2^delta, delta <= 15
```

Because `D<2^112`, the scaled residual remains below `2^127` and cannot
overflow. For `Eb>C`, both active operands are initially scaled by `2^(C-Eb)`;
inputs already below the divisor use the exact no-reduction result directly.

The stage keeps these values constant:

```text
D
rho_up(D)
D_high
D_low
```

Only `R` changes between stages. Quotient conversion and the two-component
SFPMAD proof are otherwise unchanged.

## 3. Why floating subnormals disappear

At exponent 111, the divisor's integer lattice step is at exponent 88. Thus:

- a nonzero split component is normal;
- `rho_up(D)` is normal for every normal physical divisor exponent;
- active quotient products are normal;
- nonzero component-subtraction transients are at least `2^88`;
- interstage scaling cannot underflow;
- top-range reconstruction happens in the same normalized frame.

For example, the smallest physical subnormal result `2^-149` for a divisor with
`Eb=-126` is represented internally as `2^88`. It becomes subnormal only when
the final raw pack applies the physical exponent shift `Eb-C=-237`.

## 4. Exact raw FP32 pack

The C++ and Python models implement an integer pack with this contract:

```text
pack_scaled_fp32_exact(normalized_bits, physical_exponent_shift, sign)
```

It decodes the significand and exponent, applies the exact power-of-two shift,
and constructs normal, subnormal, signed-zero, or overflow results. Subnormal
right shifts use guard/remainder information and round-to-nearest-even. Built-in
self-tests cover:

- exact normal scaling;
- smallest subnormal creation;
- signed zero;
- subnormal ties where even rounds down;
- the halfway carry from largest-subnormal range to smallest normal;
- overflow classification.

No host floating-point multiplication or store is used to form final subnormal
bits.

## 5. Physical-frame FTZ control histogram

The extended verifier records stage index, physical `K`, divisor exponent,
exact transient exponent, quotient error, first subnormal point, exact final
class, and whether the transient was negative.

For the old physical frame with divisor `0x00800000`:

```text
Passed                                      1,946,157,055
IntermediateSubnormal                         184,549,377
first point: AfterHighSubtract                 184,549,377
exact final class: Subnormal                   184,549,377
negative subnormal before correction            96,467,969
failures                                                0
```

There are no zero-, normal-result-, or intermediate-only cases in this
histogram. This aligns with the Blackhole diagnostic: the negative subset can
flush to `-0`, enter the sign-based correction, and produce an extra divisor.

## 6. Complete C++ sweeps

Every row covers all `2,130,706,432` positive normal FP32 dividends:

| Divisor bits | Value/class | Passed | Exclusions | Failures |
|---:|---|---:|---:|---:|
| `0x00800000` | smallest normal | 2,130,706,432 | 0 | 0 |
| `0x00800001` | `nextUp(smallest normal)` | 2,130,706,432 | 0 | 0 |
| `0x00800fff` | maximum low split component at `Eb=-126` | 2,130,706,432 | 0 | 0 |
| `0x00ffffff` | largest value with `Eb=-126` | 2,130,706,432 | 0 | 0 |
| `0x40400000` | 3 | 2,130,706,432 | 0 | 0 |
| `0x40a00000` | 5 | 2,130,706,432 | 0 | 0 |
| `0x40e00000` | 7 | 2,130,706,432 | 0 | 0 |
| `0x41200000` | 10 | 2,130,706,432 | 0 | 0 |
| `0x3dcccccd` | FP32 0.1 | 2,130,706,432 | 0 | 0 |
| `0x3e99999a` | FP32 0.3 | 2,130,706,432 | 0 | 0 |
| `0x41000000` | 8 | 2,130,706,432 | 0 | 0 |
| `0x40ffffff` | `nextDown(8)` | 2,130,706,432 | 0 | 0 |
| `0x41000001` | `nextUp(8)` | 2,130,706,432 | 0 | 0 |
| `0x7f7fffff` | largest normal | 2,130,706,432 | 0 | 0 |

The final-subnormal pass counts for the four `Eb=-126` divisors were:

```text
0x00800000    184,549,377
0x00800001  2,130,705,929
0x00800fff  2,129,666,543
0x00ffffff  1,019,215,872
```

The largest normal divisor now has a valid normalized reciprocal; the old
physical configuration rejected it because `1/b` was subnormal.

## 7. Independent Python cross-check

The proof-oriented Python implementation independently models the same frame,
including exact final packing. Results:

```text
deterministic + BF16 + selected random:
  tested=458214 failures=0 exclusions=0
  q_error=[0,1] max_q=65535

deterministic + 100k arbitrary normal divisors, seed 0x7a11:
  tested=100582 failures=0 exclusions=0
  q_error=[0,1] max_q=65535
```

The physical-frame model previously reported 5,580 exclusions on the second
suite: 4,063 subnormal split components, 791 non-normal reciprocals, 428 exact
subnormal intermediates/results, and 298 subnormal component subtractions.
The stationary model eliminates all four classes on the identical seed.

## 8. Files and commands

Implementations:

```text
tools/sfpi_modulo_fixed_schedule_exhaustive.cpp
tools/sfpi_modulo_fixed_schedule_reference.py
```

Primary gate:

```bash
./sfpi_modulo_fixed_schedule_exhaustive \
    --divisor-bits 0x00800000 --exhaustive-inputs \
    --exponent-stationary --working-exponent 111 --threads 16
```

Independent arbitrary-divisor check:

```bash
python3 tools/sfpi_modulo_fixed_schedule_reference.py \
    --random 100000 --skip-bf16 --random-divisors --seed 0x7a11 \
    --exponent-stationary --working-exponent 111
```

## 9. Decision and next gate

The host acceptance criterion is met, so an isolated SFPI prototype is now
justified. It should precompute normalized `D`, its biased reciprocal and split,
the stationary shift schedule, and the final physical exponent shift.

Before production integration, the following remain mandatory:

1. Implement raw normal/subnormal packing on SFPI without exceeding the eight
   visible local registers or relying on a floating store that flushes bits.
2. Perform floor-remainder sign adjustment as `D-R` in the normalized frame
   before packing; preserve dividend-signed exact zero.
3. Handle the exact `a<b` bypass for high-exponent divisors without allowing a
   masked long path to corrupt the bypassed result.
4. Compile/disassemble both architectures, reject spills, and rerun Blackhole
   raw-bit FTZ/top-range tests and performance.
5. Obtain Wormhole runtime evidence and define the special-value/zero-divisor
   operator policy.
6. Cover the complete divisor mantissa space with the planned structured
   reduced sweep before making an all-normal-divisor claim.

No SFPI or production code was changed in this host-only milestone.
