# SFPI scalar modulo raw-pack transport gate

## Result

The end-to-end test-only transport probe is complete on Wormhole B0 and
Blackhole p150b:

```text
SFPU integer constant construction
    -> LReg
    -> SFPSTORE
    -> 32-bit Dst
    -> UInt32 packer
    -> output buffer
    -> host uint32 comparison
```

The result is architecture-dependent:

| Architecture | Strongest tested raw store | Subnormal FP32 encodings |
|---|---|---|
| Blackhole | `MOD0_FMT_INT32` (`4`) | preserved bit-exactly |
| Wormhole | pre-rotated `MOD0_FMT_LO16` opaque store (`9`) | flushed to signed zero |

Both architectures preserve the tested normal encodings. The FP32 control
store flushes only subnormal encodings on both architectures, as expected.

This closes the transport gate negatively for a common exact implementation:
the exponent-stationary reducer can construct exact final subnormal bits on
Blackhole, but none of the tested Dst/packer transports can expose those bits
unchanged on Wormhole. A full reducer must not claim all-normal-input FP32
correctness on Wormhole unless a different output path is found.

## Test-only implementation

Files:

```text
tt_metal/tt-llk/tests/sources/sfpu_raw_pack_transport_test.cpp
tt_metal/tt-llk/tests/python_tests/test_sfpu_raw_pack_transport.py
```

The SFPU synthesizes one broadcast `uint32` word per row, so input unpacking
cannot destroy the value being measured. The probe includes signed zeros,
small and boundary subnormals, neighbors of the smallest normal, ordinary
normals, and maximum finite values. The output packer is configured as
`UInt32 -> UInt32`, and the host reads `numpy.uint32` words.

The host compares word multisets because SFPU lane-to-tile placement is not
part of this gate. Every synthesized row is expected 32 times.

The test deliberately records the negative Wormhole result as an architecture
contract:

- Blackhole requires exact equality to every synthesized word.
- Wormhole requires every subnormal encoding to become the corresponding
  signed zero while all other words remain exact.
- A separate FP32-store control requires the same FTZ transformation on both
  architectures.

This makes the limitation executable rather than leaving it as a prose-only
observation. If a future LLK or silicon path begins preserving the Wormhole
words, the current contract will fail and force the expectation to be reviewed.

## Wormhole isolation sequence

Several intermediate experiments were needed to isolate the conversion:

1. Reusing an unpacked UInt32 tile was rejected because the Wormhole input
   path had already flushed subnormal-looking words.
2. The final probe constructs every word directly in an LReg with SFPI integer
   constants.
3. Initializing SFPU only in the kernel prelude left later Wormhole stores
   ineffective. Re-establishing lane state after the unpack-to-Dst handshake
   with `SFPCONFIG`, the required settling `SFPNOP`, and unconditional
   `SFPENCC` made the FP32 control store behave correctly.
4. `MOD0_FMT_INT32` preserved all normal words on Wormhole but flushed all 13
   positive and negative subnormal patterns to signed zero. Blackhole preserved
   the same patterns exactly.
5. Disabling packer zero-flag substitution did not change a word, excluding
   that mechanism as the cause.
6. `MOD0_FMT_HI16` (`7`) was not a portable opaque copy in silicon testing:
   Blackhole discarded the low 16 bits.
7. Wormhole's numeric `MOD0=9` opaque store was tested with a compensating
   16-bit half rotation. Disassembly confirmed mode `9`, but the complete
   `UInt32 -> UInt32` output path still flushed every subnormal encoding.

The `InstrModLoadStore` C++ enum has no correctly named member for the
SFPSTORE-specific numeric mode `9`: its `LO16` member is value `6`, which is a
different store conversion. The test therefore names numeric `9` locally and
verifies the generated instruction.

## Compile and silicon evidence

Commands:

```bash
bash tt_metal/tt-llk/.claude/scripts/run_test.sh run \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch wormhole --test test_sfpu_raw_pack_transport.py

bash tt_metal/tt-llk/.claude/scripts/run_test.sh run \
  --worktree /home/user/tt-metal/tt_metal/tt-llk \
  --arch blackhole --test test_sfpu_raw_pack_transport.py
```

Final results:

```text
Wormhole compile  2 passed
Wormhole silicon  2 passed
Blackhole compile 2 passed
Blackhole silicon 2 passed
```

The relevant disassembly forms are:

```text
Wormhole raw:     sfpstore ...,0,9,3
Wormhole control: sfpstore ...,0,0,3   # SRCB resolves to FP32
Blackhole raw:    sfpstore ...,0,4,7
Blackhole control:sfpstore ...,0,0,7   # SRCB resolves to FP32
```

The Wormhole raw ELF also contains the point-of-use sequence:

```text
sfpconfig 15,0,1
sfpnop
sfpencc ...,2
```

## Decision

No production header is changed by this milestone, and the full stationary
reducer has not been implemented.

The next work should be split explicitly:

1. Blackhole may proceed to an isolated normalized stationary reducer and an
   integer finalizer, because its raw output gate is open.
2. Wormhole may use the same normalized reducer for performance research, but
   its contract must remain FTZ for physical subnormal results.
3. An exact cross-architecture kernel remains blocked on a new Wormhole output
   mechanism. Candidates such as split 16-bit transport or a path bypassing
   Dst/packer need their own cost and compatibility gate before reducer work is
   presented as exact.

The raw gate therefore rejects a single common exact pack variant. It supports
two architecture-specific contracts, not yet two production implementations.
