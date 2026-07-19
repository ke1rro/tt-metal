# SFPI scalar modulo stationary normalized-reducer device gate

## Result

The first test-only Blackhole exponent-stationary reducer passes its isolated
arithmetic gate on p150b silicon. It returns only the nonnegative normalized
remainder:

```text
input magnitude
    -> mandatory exponent-127 pre-half
    -> stationary fixed-schedule stages
    -> exact top-range reconstruction
    -> normalized FP32 R
```

There is no sign restoration and no physical normal/subnormal finalizer in
this milestone. For physical divisor exponent `Eb` and working exponent
`C=111`, the output contract is:

```text
R_normalized = exact_magnitude_remainder * 2**(C-Eb)
```

Every nonzero selected output is therefore an ordinary normal FP32 value. The
host compares its raw FP32 bits, not an approximate tolerance.

Final gate results:

```text
Blackhole compile  11 passed
Blackhole silicon  11 passed
```

## Test-only implementation

Files:

```text
tt_metal/tt-llk/tests/helpers/include/scalar_modulo_stationary_research.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_stationary_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_stationary.py
```

The scalar specialization precomputes:

```text
D       divisor significand at exponent 111
rho_up  nextUp(nextUp(RN32(1/D))), except exact powers of two
D_high  top 12 significand bits
D_low   D-D_high
```

Only `R` changes between schedule stages. Each active stage converts the
biased reciprocal product to `UInt16`, evaluates the two split products, and
performs one negative correction. Between stages, `R` is scaled upward by at
most 15 exponent positions. Exponent-127 inputs are halved before this process
and reconstructed as `(2R) mod D` afterward.

## Device matrix

The silicon gate uses the requested mantissa, split, exponent, quotient, and
top-range boundaries:

| Physical divisor | Description | Start shift | Initial shift | Stages |
|---:|---|---:|---:|---:|
| `0x00800000` | smallest normal | 237 | 0 | 17 |
| `0x00800001` | low-mantissa boundary | 237 | 0 | 17 |
| `0x00800fff` | split boundary | 237 | 0 | 17 |
| `0x00ffffff` | largest `Eb=-126` | 237 | 0 | 17 |
| `0x40400000` | 3 | 110 | 0 | 9 |
| `0x3dcccccd` | FP32 0.1 | 115 | 0 | 9 |
| `0x40ffffff` | `nextDown(8)` | 109 | 0 | 9 |
| `0x41000000` | 8 | 108 | 0 | 9 |
| `0x41000001` | `nextUp(8)` | 108 | 0 | 9 |
| `0x77800000` | `2^112`, large `Eb=112` | 0 | -1 | 1 |
| `0x7f7fffff` | `FLT_MAX` | 0 | -16 | 1 |

For each specialization the input builder selects all representable members
of these classes:

```text
a < b
a == b
exact power-of-two multiple
one FP32 lattice step above that multiple
floor(a/b) = 65534
floor(a/b) = 65535
ratio immediately below 65536
exponent-126/127 pre-half boundary values
positive and negative signs for the same magnitude
```

The 11 tiles contain 11,264 raw output comparisons. Their designed sets have
299 signed input patterns before tile repetition. The independent integer
oracle sees 1,000 active stages across those designed patterns; both quotient
errors `0` and `+1` occur, and every stage asserts:

```text
q_hat in {q, q+1}
```

The device does not export a separate quotient trace. The gate combines that
stage-by-stage host assertion with exact final normalized device bits; a future
diagnostic can transport individual `q_hat` values if instruction-level
quotient observation becomes necessary.

## Disassembly and register gate

All 11 MATH ELFs compile without an illegal SFPU spill. The reducer symbols
have these bounds:

```text
one-stage specializations     0xa8 bytes / 42 encoded instructions
multi-stage specializations  0xf8 bytes / 62 encoded instructions
visible local registers      L0 through L4
```

Each body has exactly:

```text
2 sfpload   # input and top-reconstruction reread
1 sfpstore  # normalized result
0 scalar lw/sw/lb/lh/sb/sh
```

Multi-stage forms contain two `ttreplay` instructions that replay the
stationary stage body. They are not spill loads/stores. The disassembly shows
the expected quotient sequence (`sfpmul`, `sfpstochrnd`, `sfpcast`), split
subtractions (`sfpsetman`, two `sfpmad`), correction, and interstage
`sfpdivp2` scaling.

## Commands

```bash
bash tt_metal/tt-llk/.claude/scripts/run_test.sh compile \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole --test test_sfpu_scalar_modulo_stationary.py --verbose

bash tt_metal/tt-llk/.claude/scripts/run_test.sh run \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole --test test_sfpu_scalar_modulo_stationary.py --verbose
```

## Decision and remaining scope

This is positive Blackhole device evidence for the normalized reduction phase.
It closes the arithmetic/register gate only for the selected normal-input and
normal-divisor matrix above. It does not yet establish a complete `fmod` or
floor-remainder operator.

The next Blackhole milestone is deliberately separate:

```text
normalized R
    -> normalized D-R adjustment for floor remainder
    -> integer round-to-nearest-even physical pack
    -> INT32 raw store
    -> UInt32 pack
```

That phase must preserve exact-zero sign, add the exact high-divisor `a<b`
bypass where normalization could underflow, and prove that reduction live
ranges end before integer packing begins. Performance and lane-mix timing also
remain unmeasured for this stationary kernel.

No production header or LLK enum is changed. Wormhole has not been compiled or
run for this reducer, and the result must not be generalized to Wormhole's
physical subnormal transport.
