# Scalar SFPI `fmod` and `remainder` Optimization

> [!WARNING]
> A subsequent correctness audit found deterministic FP32 and BF16-input
> counterexamples to the global two-correction claim at large quotient ratios.
> The measured two-correction performance results below describe a rejected or
> conditional candidate, not a globally correct implementation. See
> [SFPI_SCALAR_MODULO_CORRECTNESS_AUDIT.md](SFPI_SCALAR_MODULO_CORRECTNESS_AUDIT.md).

## 1. Scope

This document describes an optimization of the unary scalar `fmod` and
`remainder` SFPI kernels used by TT-Metal. It covers:

- the mathematical distinction between `fmod` and floor-based remainder;
- the relevant IEEE-754 binary32 properties;
- the original SFPI quotient and residual reduction algorithm;
- the generated-instruction bottleneck;
- the optimized implementation;
- disassembly and on-device performance measurements;
- numerical validation and known limitations.

The implementation was changed in both architecture trees:

```text
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/
```

For a self-contained source-level comparison, the complete Blackhole
`ckernel_sfpu_remainder.h` header is preserved in two text snapshots:

- [`source_snapshots/sfpi_remainder_before.txt`](source_snapshots/sfpi_remainder_before.txt)
- [`source_snapshots/sfpi_remainder_after.txt`](source_snapshots/sfpi_remainder_after.txt)

Compare them directly with:

```bash
diff -u \
  docs/source_snapshots/sfpi_remainder_before.txt \
  docs/source_snapshots/sfpi_remainder_after.txt
```

On-device measurements in this document were collected only on a Blackhole
P150b. The Wormhole source is kept equivalent, but this work does not claim
Wormhole runtime equivalence without a Wormhole hardware run.

## 2. Operation semantics

For a dividend `x` and a nonzero divisor `y`, both operations compute a residual
of the form

```text
r = x - q * y
```

but select the integer quotient `q` differently.

### 2.1 `fmod`

`fmod` truncates the exact quotient toward zero:

```text
q = trunc(x / y)
fmod(x, y) = x - q * y
```

Consequently:

- the result has the sign of `x`, unless it is zero;
- `abs(result) < abs(y)`;
- `fmod(-3, 2) = -1`;
- `fmod(3, -2) = 1`.

### 2.2 Floor-based remainder

The unary `remainder` operation in this code follows the framework's
floor-division remainder convention, equivalent to `torch.remainder`, rather
than the C/C++ `std::remainder` nearest-integer convention:

```text
q = floor(x / y)
remainder(x, y) = x - q * y
```

For a nonzero result, its sign follows `y`:

```text
remainder(-3,  2) =  1
remainder( 3, -2) = -1
```

This distinction is important. `std::fmod`, `std::remainder`, and
`torch.remainder` do not describe the same operation.

### 2.3 Zero divisor

Both unary SFPI implementations explicitly return quiet NaN when the scalar
divisor is zero. The zero-divisor case is handled after the ordinary arithmetic
path with a predicated replacement.

## 3. FP32 background

### 3.1 Binary32 layout

IEEE-754 binary32, commonly called FP32, contains:

| Field | Bits | Meaning |
|---|---:|---|
| Sign | 1 | Positive or negative |
| Biased exponent | 8 | Scale by a power of two |
| Fraction | 23 | Stored significand bits |

For a normal finite number, the leading significand bit is implicit. Therefore,
FP32 has 24 bits of significand precision:

```text
value = (-1)^sign * (1.fraction) * 2^(exponent - 127)
```

The spacing between adjacent normal FP32 values in the binade `[2^e, 2^(e+1))`
is

```text
ULP = 2^(e - 23)
```

Examples:

| Magnitude | Adjacent-value spacing |
|---:|---:|
| `1` | `2^-23` |
| `2^10` | `2^-13` |
| `2^22` | `0.5` |
| `2^23` | `1` |
| `2^24` | `2` |

At `2^23`, FP32 can represent every integer but not every fraction. At `2^24`,
it can represent only even integers. This matters for quotient extraction and
for tests close to exact multiples of the divisor.

### 3.2 Why reciprocal multiplication is approximate

The scalar kernels receive both the divisor and its host-computed reciprocal as
FP32 constants. Instead of a device division, they compute

```text
scaled = abs(x) * abs(1 / y)
```

Even if `x` and `y` are exactly representable, `1 / y` may not be. The reciprocal
is rounded to FP32 and the multiplication is rounded again. Therefore `scaled`
can fall on either side of the exact integer quotient near a divisibility
boundary.

For example, if the mathematical quotient is exactly `n`, the computed value may
be slightly below or above `n`. A modulo implementation must normalize the final
residual rather than assume that reciprocal multiplication identifies every
integer boundary exactly.

### 3.3 Truncating the FP32 quotient

The kernel obtains the exponent of `scaled` with `exexp`.

- If the exponent is negative, `0 <= scaled < 1`, so the truncated magnitude
  quotient is zero.
- If the exponent is below 23, fraction bits still exist. Two variable shifts
  clear the fraction bits from the FP32 bit representation.
- If the exponent is at least 23, FP32 no longer carries fractional bits at that
  magnitude, so `scaled` is already integral in its FP32 representation.

Conceptually:

```cpp
if (exp < 0) {
    quotient = 0;
} else if (exp < 23) {
    quotient = clear_fraction_bits(scaled, exp);
} else {
    quotient = scaled;
}
```

The additional comparison

```cpp
if (quotient > scaled) {
    quotient -= 1;
}
```

protects the truncation direction when the bit-manipulation path produces a
value above `scaled`.

### 3.4 Denormals

The SFPU arithmetic path used here flushes denormal values. Consequently, the
FP32 values immediately adjacent to zero do not have ordinary host binary32
modulo semantics on this path. Tests include exact zero but deliberately do not
treat the smallest host denormals as exact SFPU inputs.

## 4. Original implementation

The original hot loop had two independent costs.

### 4.1 Repeated reciprocal multiplication

The expression

```cpp
v * recip_val
```

appeared four times in source:

1. exponent extraction;
2. fraction-bit clearing;
3. the large-exponent quotient path;
4. quotient comparison.

Generated code confirmed that the compiler did not eliminate all repetitions:
the baseline `remainder` body contained four `sfpmul` instructions.

### 4.2 Ten unconditional correction blocks

After computing

```text
v = abs(x) - quotient * abs(y)
```

the original implementation emitted ten predicated corrections:

```cpp
for (int i = 0; i < 10; ++i) {
    v_if(v >= s) {
        v = s - v;
    }
    v_endif;
}
```

The loop count was compile-time constant. It did not create a cheap scalar loop;
the SFPI compiler encoded repeated vector predicate and arithmetic work. In the
measured baseline, it used `ttreplay` to represent repeated instruction
sequences compactly. `ttreplay` reduces ELF size, but the replayed instructions
still consume execution cycles.

This is why counting only static ELF instructions substantially underestimated
the performance cost.

## 5. Optimized implementation

### 5.1 Hoist the scaled quotient

The reciprocal product is now calculated once per vector step:

```cpp
const sfpi::vFloat scaled = v * recip_val;
```

All exponent, bit-clearing, selection, and comparison paths reuse `scaled`.

The generated `remainder` body changes from four `sfpmul` instructions to one.

### 5.2 Normalize the unsigned residual once in each direction

The ten correction blocks are replaced by two range-normalization predicates:

```cpp
v_if(v >= s) {
    v = v - s;
}
v_endif;

v_if(v < 0.0f) {
    v = v + s;
}
v_endif;
```

This establishes the intended unsigned residual interval around FP32 integer
boundaries before applying the operation-specific sign rule.

The transformation is deliberately described as normalization around FP32
boundaries. It is not presented as a proof that reciprocal multiplication has a
globally bounded one-unit quotient error for every finite FP32 ratio. The old
ten-step reducer was not a general proof of arbitrary-range reduction either.
The supported numerical domain is established by the operation contract and
tests.

### 5.3 Apply sign policy after normalization

After unsigned residual normalization:

- `fmod` restores the dividend sign with `copysgn`;
- floor-based `remainder` applies its dividend/divisor sign adjustments and then
  restores the divisor sign convention.

Normalizing before sign handling keeps the correction logic shared and avoids
repeated sign-sensitive correction work.

### 5.4 Unchanged data movement

This is an SFPU arithmetic optimization. It does not change:

- tile or face layout;
- destination-register traversal;
- L1/DRAM traffic;
- unpack or pack configuration;
- synchronization;
- scalar constant programming.

The expected gain is therefore isolated to math execution rather than memory
movement.

## 6. Generated instruction analysis

The ELF was disassembled with the pinned SFPI toolchain:

```bash
riscv-tt-elf-objdump -d -C math.elf
```

For `calculate_remainder<false, 32>()`:

The checked-in symbol dumps compare the original baseline with the final safe
caching-only implementation. They do not represent the rejected two-correction
candidate discussed in the performance table below:

- [`disassembly/sfpi_remainder_before.disasm`](disassembly/sfpi_remainder_before.disasm)
- [`disassembly/sfpi_remainder_after.disasm`](disassembly/sfpi_remainder_after.disasm)

They can be compared without filtering a complete kernel ELF:

```bash
diff -u \
  docs/disassembly/sfpi_remainder_before.disasm \
  docs/disassembly/sfpi_remainder_after.disasm
```

| Instruction or group | Baseline | Optimized | Difference |
|---|---:|---:|---:|
| Encoded function instructions | 66 | 59 | -7 |
| `sfpmul` | 4 | 1 | -3 |
| `sfpmad` | 4 | 3 | -1 |
| `ttreplay` | 5 | 0 | -5 |
| `sfpmov` | 5 | 7 | +2 |

The table immediately above is retained as the disassembly accounting for the
rejected two-correction candidate. The final caching-only files linked above
instead contain 66 versus 65 encoded instructions, `sfpmul` 4 versus 1, and
retain five `ttreplay` instructions.

The total static count falls by seven instructions. The dynamic saving is much
larger because each baseline `ttreplay` expands into repeated work at execution
time.

Math-isolate ELF text size also decreased:

| Operation | Baseline | Optimized | Difference |
|---|---:|---:|---:|
| `fmod` | 2860 bytes | 2804 bytes | -56 bytes (-1.96%) |
| `remainder` | 2940 bytes | 2884 bytes | -56 bytes (-1.90%) |

## 7. Blackhole performance measurement

### 7.1 Method

The on-silicon LLK benchmark configuration was:

| Parameter | Value |
|---|---|
| Device | Blackhole P150b |
| Input/output | FP32 to FP32 |
| Destination accumulation | Enabled |
| SFPU iterations | 32 |
| Tile count | 8 |
| Loop factor | 16 |
| Measurement zone | `MATH_ISOLATE` |

`MATH_ISOLATE` measures the math body independently of unpack and pack. Program
construction is also outside the measured tile loop.

The baseline and optimized kernels were built and run with the same checkout,
device, test harness, formats, and toolchain. Only the two Blackhole scalar
kernel headers were temporarily returned to baseline for the baseline run; the
optimized implementation was restored immediately afterward.

### 7.2 Results

| Operation | Baseline cycles/tile | Optimized cycles/tile | Cycles removed | Reduction | Speedup |
|---|---:|---:|---:|---:|---:|
| `fmod` | 2976.242 | 1568.227 | 1408.016 | 47.31% | 1.898x |
| `remainder` | 3328.242 | 1888.227 | 1440.016 | 43.27% | 1.763x |

The measured dynamic result, rather than the static instruction count, is the
authoritative performance number.

These results describe the isolated SFPU math stage. End-to-end operator
speedup will be smaller when unpack, pack, dispatch, or memory movement is a
significant fraction of total runtime.

## 8. Correctness validation

### 8.1 Domain tests

The existing unary SFPU domain matrix covers both operations across supported
FP32 and BF16 input/output combinations and both destination accumulation modes.
On the Blackhole system used for this work:

```text
10 supported domain cases passed
6 unsupported harness combinations skipped
```

The six skips are pre-existing Blackhole LLK harness restrictions. With
destination accumulation disabled, only FP32-to-FP32 is supported in this test
path; the three other format combinations are skipped for each of the two
operations.

### 8.2 Exact boundary tests

A dedicated test constructs values around modulo boundaries using exact FP32
values and `torch.nextafter`. It covers:

- positive and negative zero where representable by the operation path;
- positive and negative dividends;
- fractions and nonmultiples;
- exact multiples of the fixed scalar divisor `2.0`;
- adjacent FP32 values below and above those boundaries;
- powers from `2^-10` through `2^24`;
- the `2^23` boundary where FP32 stops representing fractional units;
- both destination accumulation modes;
- both `fmod` and floor-based `remainder`.

The test uses:

```text
atol = 0
rtol = 0
```

Across the two operations, it exercises 208 explicitly constructed boundary
inputs and requires exact equality.

The wider `nextafter` exponent sweep runs with destination accumulation enabled.
When destination accumulation is disabled, the Blackhole LLK harness can narrow
the input during unpack, so arbitrary FP32 adjacent values cannot all reach the
SFPU unchanged. That mode uses a smaller set of exactly representable boundaries
instead of hiding input conversion behind a loose tolerance.

### 8.3 Test result

The combined focused run after formatting and optimization produced:

```text
14 passed, 6 skipped
```

## 9. Reproduction

### 9.1 Correctness

```bash
cd tt_metal/tt-llk/tests/python_tests
unset ARCH_NAME

python3 -m pytest test_sfpu_unary.py -q \
  -k 'modulo_boundaries or ((Remainder or Fmod) and test_eltwise_unary_sfpu_domain)'
```

### 9.2 Performance

`MathOperation.Fmod` and `MathOperation.Remainder` are included in
`perf_eltwise_unary_sfpu.py`. A focused node can be run directly, for example:

```bash
python3 -m pytest -q -s \
  'perf_eltwise_unary_sfpu.py::test_perf_eltwise_unary_sfpu[formats:Float32->Float32-approx_mode:No-mathop:Fmod-dest_acc:Yes-loop_factor:16-iterations:32-fast_mode:No-stable_sort:No-input_dimensions:[128, 64]]'
```

The postprocessed result is written under:

```text
tt_metal/tt-llk/perf_data/perf_eltwise_unary_sfpu/
```

Use the `TILE_LOOP` row and `mean(MATH_ISOLATE)` column for cycles per tile.

### 9.3 Disassembly

The LLK tests build `math.elf` files below `/tmp/tt-llk-build`. Locate the ELF
containing the desired symbol and disassemble it with:

```bash
tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-nm -C math.elf \
  | grep calculate_remainder

tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-objdump -d -C math.elf
```

For production TT-Metal kernels, `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1` can be
used to expose the generated ELF path. `TT_METAL_RISCV_DEBUG_INFO=1` enables
source-annotated `objdump -S` output.

## 10. Limitations and follow-up work

- Runtime measurements currently cover Blackhole only.
- The divisor in the unary LLK harness is fixed to `2.0`; production scalar
  tests should additionally sweep diverse normal divisors and their reciprocals.
- Zero-divisor NaN and infinity behavior should remain covered at the TTNN API
  level in addition to LLK tests.
- End-to-end TTNN benchmarks are needed to quantify how much of the isolated
  1.76x-1.90x math speedup survives dispatch, unpack, pack, and memory costs.
- A same-input Wormhole run is required before claiming architecture-equivalent
  numerical output or speedup.

## 11. Summary

The optimization removes repeated reciprocal products and replaces a fixed
ten-step predicated correction sequence with two residual-normalization steps.
It preserves the kernel's dataflow and FP32 arithmetic while substantially
reducing SFPU execution work.

On Blackhole, isolated math improved from 2976 to 1568 cycles/tile for `fmod`
and from 3328 to 1888 cycles/tile for floor-based `remainder`. Exact boundary
tests and the supported format-domain matrix pass after the change.
