# Blackhole FastBounded Exact Scalar Modulo Research

## Result

A test-only `FastBoundedExactBH` kernel now passes the selected Blackhole
end-to-end gate for scalar `fmod` and floor-style remainder. It has an explicit
pre-launch quotient contract, two short reciprocal/truncated-quotient reduction
stages, and the already silicon-proven exact Blackhole finalizer.

This is not a production change and it is not the historical one-shot
two-correction candidate. The one-shot body is not bit-exact on Blackhole even
inside the proposed quotient range because a large `q*b` subtraction can lose
remainder bits in SFPMAD's partially fused product. The accepted body bounds
each local product instead.

Final selected evidence:

```text
Host contract/proof-pattern audit     passed
Blackhole compile                     89/89 pytest items
Blackhole silicon                     89/89 pytest items
Device specializations                88
Raw UInt32 words compared             90,112
Disassembled exact symbols            88/88
Explicit SFPU locals                  L0-L7 maximum
Input sfpload / final raw sfpstore    1 / 1 per exact symbol
Scalar memory / out-of-line calls     0 / 0
MATH_ISOLATE performance              32/32 passed on Blackhole
Production headers                    unchanged
Git state                             FastBounded work intentionally uncommitted
```

## Scope and unresolved policy

The selected contract accepts zero or finite normal FP32 dividend magnitudes
and a finite normal scalar divisor magnitude. It covers both operand signs,
`fmod`, floor remainder, exact zero, normal output, and exact Blackhole
subnormal output.

It does not define NaN, infinity, zero-divisor, subnormal-input, tensor-wide
range discovery, or production dispatch policy. The selected matrix is device
evidence, not an exhaustive proof over every FP32 pair.

This implementation is Blackhole-only. It uses Blackhole FP32-to-UInt16
round-toward-zero conversion and the Blackhole-proven integer/raw final
transport. Wormhole does not expose the same conversion form, and the tested
Wormhole raw transport flushes subnormal results. A Wormhole port therefore
needs a different truncation and final transport, or an explicit FTZ contract.

## Why the historical one-shot body was not accepted

The attractive source shape was:

```text
scaled = RN32(A * reciprocal)
q_hat  = trunc(scaled)
R      = A - q_hat * D
correct R by at most one D
```

The `scaled` exponent contract can bound the integer quotient error, but it
does not make the single large product/subtraction bit-exact on Blackhole.
Direct device/model counterexamples found during this milestone include:

```text
a = 0x4b46b792, b = 0x4046beae
q_hat = q + 1
corrected one-shot result = 0x403ebeae
exact result              = 0x403738fa
```

Even using an exact quotient and splitting the divisor more aggressively did
not make the large intermediate subtraction generally exact:

```text
a = 0x4b5df984, b = 0x40687b64
split-product result = 0x3d119700
exact result         = 0x3d11a700
```

The global RN estimate can also be one below the exact quotient:

```text
a = 0x5e6e2232, b = 0x537f6bdb
q_hat - q = -1
```

A different two-ULP global reciprocal experiment produced `q_hat-q=+2` near
the boundary. Therefore neither the old measured body nor its
`1568/1888 cycles/tile` result is a correctness baseline for this exact
milestone.

## Explicit caller contract

Let `Eb` be the unbiased exponent of the positive normal scalar divisor. The
host/planner constructs:

```text
D103 = exact exponent scaling of abs(b) to exponent 103
A103 = exact exponent scaling of abs(a) by 103-Eb
rho  = RN32(1 / D103)
s    = RN32(A103 * rho)
```

For `abs(a) >= abs(b)`, the selected FastBounded contract is:

```text
s is finite
and
(s == 0 || exexp(s) < 22)
```

For `abs(a) < abs(b)`, reduction is unnecessary and the direct bypass is safe.
The caller must select FastBounded before launch. There is no lane-local test
and no masked Robust fallback in this kernel.

This is deliberately not stated as `abs(a) < 2^22`. The useful bound is on the
actual scaled ratio estimate for the specialized divisor. For example, the
last accepted and first rejected positive FP32 input words for `b=3` are:

```text
last accepted   0x4b3fffff
first rejected  0x4b400000
```

The host audit observes all three permitted global estimate errors:

```text
q_hat - floor(abs(a)/abs(b)) in {-1, 0, +1}
```

The global estimate is a contract witness. It is not used as the actual large
reduction quotient.

## Accepted two-stage exact reducer

The reducer works in the exponent-103 divisor lattice and uses the normalized
exponent-111 divisor already required by the finalizer:

```text
D111 = 256 * D103
rho_up = rho                         for a power-of-two divisor
rho_up = raw_next(raw_next(rho))     otherwise

R = A103
R = local_stage(R, D111, rho_up / 256)
R = local_stage(R, D103, rho_up)
R111 = R * 256
```

Each active local stage performs:

```text
q_local = trunc_to_UInt16_RTZ(RN32(R * local_reciprocal_up))
R       = R - q_local * D_high
R       = R - q_local * D_low
if R < 0:
    R += D
```

`D_high` and `D_low` each have at most 12 significant bits. The first exact
local quotient is at most `2^14`, and its one-sided estimate is at most
`2^14+1`. The second exact quotient is below `2^8`, and its estimate is at
most `2^8`. Both estimates fit UInt16, and every quotient/component product
fits the established 28-bit SFPMAD product-precision shape. The exact host
integer model observes only:

```text
q_local_hat - q_local in {0, +1}
```

so one negative correction restores the exact divisor-lattice remainder. The
final `*256` is an exact exponent change into the exponent-111 finalizer
contract.

The source uses `convert<vUInt16>(..., RoundMode::Zero)`. On final Blackhole
lowering this is represented by the expected conversion instruction sequence;
the silicon raw-bit result, rather than source syntax alone, closes its semantic
gate.

## Finalization and sign semantics

The accepted reducer reuses the committed stationary combined helpers:

```text
normalized R111
    -> exact-zero test
    -> optional D111-R111 for different-sign floor remainder
    -> exact normal/subnormal integer RNE pack
    -> fmod or divisor sign restoration
    -> one raw INT32 store
    -> UInt32 pack
```

For `Eb<=111`, an `a<b` input is normalized directly and passed to the same
finalizer. For `Eb>111`, the committed physical-frame bypass is reused so a
small input cannot underflow while being scaled down. Exact-zero is tested
before `D-R`, so an exact multiple remains zero rather than becoming `D`.

## Host and silicon matrix

The correctness test compiles both operations and all four operand-sign
combinations for eleven divisor specializations:

| Divisor | Purpose |
|---|---|
| `3.0f` | ordinary normal result and typical scalar |
| `0x00800000` | smallest normal divisor; full subnormal pack |
| `0x00800001` | difficult lattice boundaries |
| `Eb=-103` | normal-only output boundary |
| `Eb=-104` | first subnormal-capable class |
| `Eb=111` | high working-frame specialization |
| `Eb=112` | physical bypass path |
| `FLT_MAX` | top-range/bypass path |
| `0x4046beae` | rejected one-shot product regression |
| `0x40687b64` | rejected component-subtraction regression |
| `0x537f6bdb` | global underestimate/boundary regression |

Generated patterns cover `a<b`, equality, exact multiples, adjacent FP32
values around multiples, quotient targets `0`, `1`, `2`, `65535`, values near
`2^21` and `2^22`, and the actual last-safe/first-unsafe contract boundary.
Unsafe boundary values are rejected by the host contract and are not sent to
the Fast kernel.

The persistent test contains 287 divisor/magnitude patterns and 2,296 logical
operation/sign expansions. Each of the 88 device specializations compares one
full 1024-word tile, for 90,112 raw UInt32 comparisons. Coverage assertions
require exact zero, normal output, a general subnormal, smallest subnormal
`0x00000001`, and largest subnormal `0x007fffff`.

The host compares output-word multisets because FP32 input and UInt32 output
tile lane placement is outside this arithmetic/raw-transport gate. It detects
every missing, extra, or incorrectly encoded word, but a future production
gate must additionally use a lane-aware mapping.

Supplemental review sweeps found no blocker in 190,028 random safe FP32 pairs,
and 143,146 random pairs passed both split stages through the Blackhole ISA FMA
model. These are additional evidence, not a replacement for a formal full-pair
proof.

Final commands:

```bash
bash tt_metal/tt-llk/.claude/scripts/run_test.sh run \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole \
  --test test_sfpu_scalar_modulo_fast_bounded.py --verbose
```

Result:

```text
compile:  89 passed
silicon:  89 passed
```

## Disassembly and register gate

The audit isolates the exact templated function by its `nm` address and size,
then disassembles only that body. All 88 correctness ELFs report:

```text
88 symbols found
88 with exactly one sfpload L?,0,0,7
88 with exactly one sfpstore L?,0,4,7
88 with zero lw/sw/lb/lh/lbu/lhu/sb/sh/flw/fsw
88 with zero out-of-line calls or emitted Robust fallback symbols
88 with no L8/L10 or unexpected local register
maximum explicit local L7
symbol size 0xec-0x1d8 bytes (59-118 instructions)
```

The observed register set is `L0-L7` plus architectural/programmed constants
`L9` and `L11-L14`. The two local stages lower with schedule replays; these are
not memory spills.

All eight `Eb=-104` operation/sign forms retain the required first classifier
shape:

```text
sfpexexp ...
sfpiadd ...,0xFA7,1
```

The later `0xFA7,5` normal-path instruction is expected. The correctness gate
is that the first classifier occurrence after `sfpexexp` uses LT/MOD1=1; the
rejected GTE-first selector did not return.

## Performance

Fast and committed combined Robust were compiled in one test source with only a
compile-time path selector. Both use eight tiles, loop factor 16, four faces,
FP32 destination accumulation, `unpack_to_dest=false`, and `MATH_ISOLATE`.
Fmod uses positive operands; floor remainder uses positive dividend and
negative divisor so the `D-R` path is present.

All 32 specializations compiled and ran on Blackhole:

| Operation | Divisor | Fast | Robust | Robust/Fast |
|---|---|---:|---:|---:|
| fmod | `3.0` | 4027.430 | 7963.688 | 1.977x |
| fmod | smallest normal | 4027.445 | 12827.672 | 3.185x |
| fmod | smallest normal + 1 ULP | 4027.430 | 12827.672 | 3.185x |
| fmod | `Eb=-103` | 4027.445 | 12219.672 | 3.034x |
| fmod | `Eb=-104` | 4027.445 | 12219.672 | 3.034x |
| fmod | `Eb=111` | 2587.422 | 1627.422 | 0.629x |
| fmod | `Eb=112` | 2491.445 | 2171.422 | 0.872x |
| fmod | `FLT_MAX` | 2491.430 | 2171.430 | 0.872x |
| floor | `3.0` | 4379.430 | 8347.461 | 1.906x |
| floor | smallest normal | 4379.422 | 13211.438 | 3.017x |
| floor | smallest normal + 1 ULP | 4379.430 | 13211.445 | 3.017x |
| floor | `Eb=-103` | 4379.422 | 12603.438 | 2.878x |
| floor | `Eb=-104` | 4379.422 | 12603.438 | 2.878x |
| floor | `Eb=111` | 2939.422 | 2011.422 | 0.684x |
| floor | `Eb=112` | 3003.445 | 2683.430 | 0.893x |
| floor | `FLT_MAX` | 3003.430 | 2683.469 | 0.893x |

The exact Fast body is effectively fixed-cost for the low/mid exponent classes
and is 1.91-3.19x faster than Robust in this matrix. For `Eb>=111`, Robust has
only a very short schedule and is 12-59% faster than Fast. Therefore a future
planner should consider both proof-safe input range and divisor specialization;
“contract safe” does not automatically mean “Fast is cheaper.”

The historical `1568/1888 cycles/tile` measurements omitted the accepted exact
two-stage reduction/finalization shape and belonged to a rejected correctness
candidate. They must not be advertised as the cost of `FastBoundedExactBH`.

## Decision

Architecture B now has a selected, silicon-proven test-only pair on Blackhole:

```text
FastBoundedExactBH
    caller proves the actual scaled quotient contract
    two short exact local stages
    2491-4379 cycles/tile in the measured matrix

RobustExactBH
    no bounded-quotient assumption in its documented normal domain
    exponent-dependent stationary schedule
    1627-13211 cycles/tile in the measured matrix
```

They must remain separate compiled kernels. No tensor scan or lane-local Robust
fallback was added. Production integration should wait for API/range-metadata
review, special-value policy, a lane-aware output gate, and an independent
Wormhole contract/implementation.
