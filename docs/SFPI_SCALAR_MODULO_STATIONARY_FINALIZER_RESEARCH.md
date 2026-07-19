# SFPI scalar modulo stationary finalizer device gate

## Result

The isolated test-only Blackhole stationary-result finalizer passes its raw-bit
correctness and register gates on p150b silicon:

```text
positive normalized R
    -> optional normalized D-R
    -> integer physical exponent conversion
    -> integer RNE normal/subnormal encoding
    -> fmod or floor-remainder sign restoration
    -> INT32 SFPSTORE
    -> UInt32 pack
```

Final results:

```text
Blackhole compile  13/13 passed
Blackhole silicon  13/13 passed
Raw comparisons    13,312
```

The expected words come from an independent integer-lattice oracle. The gate
does not use NumPy or PyTorch floating arithmetic to calculate a remainder or
to scale the expected result.

This is still an isolated phase result. The normalized reducer and finalizer
have not yet been combined in one SFPU symbol, and no performance claim is
made.

## Test-only implementation

Files:

```text
tt_metal/tt-llk/tests/helpers/include/scalar_modulo_stationary_finalizer_research.h
tt_metal/tt-llk/tests/sources/sfpu_scalar_modulo_stationary_finalizer_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_scalar_modulo_stationary_finalizer.py
```

The kernel accepts a positive normal FP32 `R` in the reducer's exponent-111
frame. The scalar divisor exponent and operation/sign choices are compile-time
parameters. The only programmable float constant is normalized `D`, used by
the floor-remainder `D-R` adjustment.

No production SFPI header or LLK enum is changed.

## Exact-zero and sign order

The finalizer evaluates the exact-zero predicate on the original reducer
output. For floor remainder with different operand signs, it performs `D-R`
only when that original `R` is nonzero:

```text
exact R == 0: preserve zero magnitude
same signs:   magnitude = R
different signs and R != 0: magnitude = D-R
```

Performing `D-R` in the normalized frame keeps the subtraction away from the
physical subnormal/FTZ range. Sign restoration occurs only after the raw
positive encoding has been constructed:

```text
fmod nonzero or zero          dividend sign
floor remainder nonzero      divisor sign
floor remainder exact zero   dividend sign
```

The four compile-time sign combinations are tested for both operations. In
particular, the different-sign floor cases prove that exact zero does not turn
into `D`, while nonzero results use the divisor sign.

## Integer FP32 pack

For physical divisor exponent `Eb`, normalized values are scaled by the exact
power of two `2**(Eb-111)`. The finalizer does not execute that scale as FP32
arithmetic. It extracts the unbiased normalized exponent and computes:

```text
min_normalized_exponent = -126 - (Eb-111)
result_biased_exponent  = normalized_exponent
                          - min_normalized_exponent + 1
```

For a normal physical result, exponent and mantissa are assembled separately:

```text
(result_biased_exponent << 23) | normalized_mantissa
```

This avoids both a physical subnormal intermediate and an integer addition
across the mantissa/exponent boundary.

For a subnormal physical result, the normal 24-bit significand is shifted
right. The discarded part and halfway value are constructed without a
`1 << shift` expression:

```text
truncated = significand >> shift
discarded = significand - (truncated << shift)
half      = 0x00800000 >> (24-shift)
```

The increment is:

```text
discarded > half
or (discarded == half and retained LSB is odd)
```

`shift == 24` includes the tie-to-zero boundary. A shift above 24 is strictly
below halfway for a 24-bit significand, so the zero-initialized output remains
zero and no out-of-range host/device shift is issued. Rounding can carry
`0x007fffff` to the smallest normal encoding `0x00800000` naturally.

## Device matrix

The 13 specializations are:

| Cases | Divisor | `Eb` | Purpose |
|---:|---:|---:|---|
| 4 | `0x00800000` | -126 | fmod, all operand-sign combinations, full RNE pack |
| 4 | `0x00800001` | -126 | floor remainder, all signs, exact zero and normalized `D-R` |
| 1 | `0x0c000000` | -103 | normal-only physical boundary |
| 1 | `0x0b800000` | -104 | first full-pack physical boundary |
| 1 | `0x77000000` | 111 | no physical exponent downscale |
| 1 | `0x77800000` | 112 | positive physical exponent shift |
| 1 | `0x7f7fffff` | 127 | largest finite divisor exponent/mantissa |

The smallest-normal fmod inputs explicitly exercise:

```text
zero
smallest normal
nextUp(smallest normal)
largest subnormal
smallest subnormal
below halfway
halfway with retained LSB even
halfway with retained LSB odd
above halfway
rounding carry to smallest normal
underflow tie to zero
underflow above tie to 0x00000001
shift above 24
```

The floor matrix includes results where normalized `D-R` becomes physical
normal, largest-subnormal, and smallest-subnormal encodings. The high-exponent
cases exercise the short normal-only code shape.

Each specialization processes a complete 32x32 tile. The designed sets contain
97 case/input patterns before tile repetition, for 13,312 raw device words.
The host compares word multisets because SFPU lane-to-tile placement is outside
this arithmetic/transport gate; count mismatches include positional samples for
diagnosis.

## Blackhole lowering finding

Two initial normal/subnormal selectors used a greater-than-or-equal form: first
an FP32 comparison with a programmable threshold, then an integer exponent
comparison. In this nested predicated code shape, neither selected the expected
normal lanes on p150b. A sentinel run isolated the failure to the selector;
the candidate `addexp`/`setexp` normal encoders were not executed for those
lanes and therefore were not the demonstrated cause.

The passing form makes the subnormal case the primary branch:

```text
if normalized_exponent < min_normalized_exponent:
    integer RNE subnormal pack
else:
    integer normal encoding
```

It lowers through the basic less-than/sign condition and passes the exact
`Eb=-103` and `Eb=-104` silicon boundaries. The final implementation also
constructs normal output bits directly rather than relying on FP exponent
transport.

## Disassembly and register gate

All 13 MATH ELFs compile without an illegal SFPU spill. The finalizer symbol
bounds are:

| Code shape | Bytes | Encoded instructions | Visible explicit locals |
|---|---:|---:|---|
| high-exponent normal-only | `0x48` | 18 | `L0-L2` |
| unsigned/full RNE pack | `0xfc` | 63 | `L0-L6` |
| fmod sign restoration | `0x104` | 65 | `L0-L6` |
| floor `D-R` and sign restoration | `0x124-0x128` | 73-74 | `L0-L7` |

`L9` is the architectural zero constant and `L12` is the programmed normalized
divisor constant; neither is an allocated explicit local. Every symbol has:

```text
1 sfpload
1 sfpstore ...,0,4,7
0 scalar lw/sw/lb/lh/sb/sh
0 unexpected Dst reloads
```

The `MOD0=4` store is the Blackhole INT32 raw transport already proven by the
separate raw-pack gate. The output packer is explicitly configured as UInt32.

## Commands

```bash
bash tt_metal/tt-llk/.claude/scripts/run_test.sh compile \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole \
  --test test_sfpu_scalar_modulo_stationary_finalizer.py \
  --verbose

bash tt_metal/tt-llk/.claude/scripts/run_test.sh simulate \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole \
  --test test_sfpu_scalar_modulo_stationary_finalizer.py \
  --verbose
```

The runner's `simulate` phase consumes the compiled ELFs on the locked
Blackhole hardware; it is not software-only arithmetic evidence.

## Decision and remaining scope

This closes the isolated Blackhole finalizer correctness, raw-transport, and
register gate for the selected normalized inputs. The next milestone is to
combine it with the stationary reducer and prove that reducer live ranges end
before the pack phase begins.

Still unresolved:

1. Combined and end-to-end smallest-normal, top-range, and `a<b` bypass cases.
2. Special values, zero divisor, subnormal inputs, and operator-level FTZ
   policy.
3. Finalizer, combined-kernel, and lane-mix performance.
4. The later normal-only/full-pack specialization split.
5. Wormhole exact output, which remains blocked by its measured subnormal
   transport behavior.
6. Production API, dispatch, and integration review.

No claim here extends to all FP32 operand pairs or to Wormhole silicon.
