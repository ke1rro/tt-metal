# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Raw-bit Blackhole gate for the isolated stationary-result finalizer."""

import struct
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pytest
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import BlocksCalculationAlgorithm, DestAccumulation, DestSync
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)

WORKING_EXPONENT = 111
MANTISSA_MASK = 0x007FFFFF
SIGN_MASK = 0x80000000


class ScalarModuloKind(IntEnum):
    FMOD = 0
    FLOOR_REMAINDER = 1


@dataclass(frozen=True)
class FinalizerCase:
    divisor_bits: int
    kind: ScalarModuloKind
    dividend_negative: bool
    divisor_negative: bool
    name: str


FINALIZER_CASES = [
    *[
        FinalizerCase(
            0x00800000,
            ScalarModuloKind.FMOD,
            dividend_negative,
            divisor_negative,
            f"fmod_min_normal_a{'neg' if dividend_negative else 'pos'}_b{'neg' if divisor_negative else 'pos'}",
        )
        for dividend_negative in (False, True)
        for divisor_negative in (False, True)
    ],
    *[
        FinalizerCase(
            0x00800001,
            ScalarModuloKind.FLOOR_REMAINDER,
            dividend_negative,
            divisor_negative,
            f"floor_boundary_a{'neg' if dividend_negative else 'pos'}_b{'neg' if divisor_negative else 'pos'}",
        )
        for dividend_negative in (False, True)
        for divisor_negative in (False, True)
    ],
    FinalizerCase(
        0x0C000000, ScalarModuloKind.FMOD, False, False, "normal_only_eb_minus_103"
    ),
    FinalizerCase(
        0x0B800000, ScalarModuloKind.FMOD, False, False, "full_pack_eb_minus_104"
    ),
    FinalizerCase(0x77000000, ScalarModuloKind.FMOD, False, False, "eb_111"),
    FinalizerCase(0x77800000, ScalarModuloKind.FMOD, False, False, "eb_112"),
    FinalizerCase(0x7F7FFFFF, ScalarModuloKind.FMOD, False, False, "flt_max"),
]


def _bits(value: float | np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _normal_parts(bits: int) -> tuple[int, int, int]:
    biased = (bits >> 23) & 0xFF
    assert 0 < biased < 0xFF
    exponent = biased - 127
    return (1 << 23) | (bits & MANTISSA_MASK), exponent - 23, exponent


def _round_shift_right_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    truncated = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return truncated + int(
        remainder > halfway or (remainder == halfway and (truncated & 1))
    )


def _compose_rne_bits(significand: int, unit_exponent: int) -> int:
    if significand == 0:
        return 0
    top_exponent = unit_exponent + significand.bit_length() - 1
    if top_exponent < -126:
        subnormal = _round_shift_right_even(significand, -149 - unit_exponent)
        return 0x00800000 if subnormal >= 0x00800000 else subnormal

    precision_shift = significand.bit_length() - 24
    normalized = _round_shift_right_even(significand, precision_shift)
    if normalized == 1 << 24:
        normalized >>= 1
        top_exponent += 1
    assert top_exponent <= 127
    return ((top_exponent + 127) << 23) | (normalized & MANTISSA_MASK)


def _scale_and_pack_rne(normalized_bits: int, exponent_shift: int) -> int:
    if normalized_bits == 0:
        return 0
    significand, unit_exponent, _ = _normal_parts(normalized_bits)
    return _compose_rne_bits(significand, unit_exponent + exponent_shift)


def _normalized_divisor_bits(divisor_bits: int) -> int:
    _, _, divisor_exponent = _normal_parts(divisor_bits)
    biased = WORKING_EXPONENT + 127
    return (biased << 23) | (divisor_bits & MANTISSA_MASK)


def _exact_positive_difference_bits(lhs_bits: int, rhs_bits: int) -> int:
    lhs_m, lhs_unit, _ = _normal_parts(lhs_bits)
    rhs_m, rhs_unit, _ = _normal_parts(rhs_bits)
    common_unit = min(lhs_unit, rhs_unit)
    lhs_integer = lhs_m << (lhs_unit - common_unit)
    rhs_integer = rhs_m << (rhs_unit - common_unit)
    assert lhs_integer > rhs_integer
    return _compose_rne_bits(lhs_integer - rhs_integer, common_unit)


def _expected_bits(normalized_r_bits: int, case: FinalizerCase) -> int:
    _, _, divisor_exponent = _normal_parts(case.divisor_bits)
    magnitude_bits = normalized_r_bits
    signs_differ = case.dividend_negative != case.divisor_negative
    if (
        case.kind == ScalarModuloKind.FLOOR_REMAINDER
        and signs_differ
        and normalized_r_bits != 0
    ):
        magnitude_bits = _exact_positive_difference_bits(
            _normalized_divisor_bits(case.divisor_bits), normalized_r_bits
        )

    packed = _scale_and_pack_rne(magnitude_bits, divisor_exponent - WORKING_EXPONENT)
    if normalized_r_bits == 0 or case.kind == ScalarModuloKind.FMOD:
        negative = case.dividend_negative
    else:
        negative = case.divisor_negative
    return packed | (SIGN_MASK if negative else 0)


def _fmod_rne_inputs(case: FinalizerCase) -> list[int]:
    if case.divisor_bits == 0x00800000:
        return [
            0x00000000,
            0x77000000,  # smallest normal physical result
            0x77000001,  # nextUp(smallest normal)
            0x76FFFFFE,  # largest subnormal
            0x6B800000,  # smallest subnormal
            0x75800003,  # below halfway
            0x75800004,  # halfway, retained LSB even
            0x7580000C,  # halfway, retained LSB odd
            0x75800005,  # above halfway
            0x76FFFFFF,  # rounding carry to smallest normal
            0x6B000000,  # underflow tie to zero
            0x6B000001,  # underflow above tie to one
            0x6A800000,  # shift > 24, strictly below halfway
        ]

    normalized_divisor = _normalized_divisor_bits(case.divisor_bits)
    return [
        0x00000000,
        0x6B800000,
        0x6C000000,
        normalized_divisor - 1,
        normalized_divisor,
    ]


def _floor_inputs(case: FinalizerCase) -> list[int]:
    normalized_divisor = _normalized_divisor_bits(case.divisor_bits)
    values = [
        0x00000000,
        0x6B800000,  # one physical subnormal lattice step
        0x6C000000,  # two lattice steps
        0x76FFFFFE,  # largest physical subnormal
        0x77000000,  # D minus one lattice step
    ]
    assert all(value == 0 or value < normalized_divisor for value in values)
    return values


def _input_bits(case: FinalizerCase) -> list[int]:
    if case.kind == ScalarModuloKind.FLOOR_REMAINDER:
        return _floor_inputs(case)
    return _fmod_rne_inputs(case)


@dataclass
class FinalizerConstants(TemplateParameter):
    case: FinalizerCase

    def convert_to_cpp(self) -> str:
        _, _, divisor_exponent = _normal_parts(self.case.divisor_bits)
        return "\n".join(
            (
                f"constexpr std::uint32_t SCALAR_NORMALIZED_DIVISOR = {_normalized_divisor_bits(self.case.divisor_bits)}u;",
                f"constexpr int SCALAR_DIVISOR_EXPONENT = {divisor_exponent};",
                f"constexpr int FINALIZER_KIND = {int(self.case.kind)};",
                f"constexpr bool FINALIZER_DIVIDEND_NEGATIVE = {str(self.case.dividend_negative).lower()};",
                f"constexpr bool FINALIZER_DIVISOR_NEGATIVE = {str(self.case.divisor_negative).lower()};",
            )
        )


def _run(case: FinalizerCase) -> tuple[np.ndarray, np.ndarray]:
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("the exact raw stationary finalizer is Blackhole-only")

    formats = InputOutputFormat(DataFormat.Float32, DataFormat.UInt32)
    input_dimensions = [32, 32]
    values = np.asarray(_input_bits(case), dtype=np.uint32).view(np.float32)
    src_a, tile_count_a, src_b, tile_count_b = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.custom(values.astype(float).tolist()),
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DestAccumulation.Yes,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    configuration = TestConfig(
        "sources/sfpu_scalar_modulo_stationary_finalizer_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            FinalizerConstants(case),
        ],
        runtimes=[
            TILE_COUNT(tile_count_a),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_a,
            formats.input_format,
            src_b,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count_a,
            tile_count_B=tile_count_b,
            tile_count_res=tile_count_a,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )
    result = np.asarray(configuration.run().result, dtype=np.uint32)
    return result, np.asarray(src_a, dtype=np.float32)


@pytest.mark.parametrize("case", FINALIZER_CASES, ids=lambda case: case.name)
def test_scalar_modulo_stationary_finalizer_raw_bits(case):
    result, source = _run(case)
    source_bits = source.view(np.uint32).reshape(-1)
    expected = np.asarray(
        [_expected_bits(int(bits), case) for bits in source_bits], dtype=np.uint32
    )
    result_bits = result.reshape(-1)
    observed_counts = Counter(int(bits) for bits in result_bits)
    expected_counts = Counter(int(bits) for bits in expected)
    mismatches = np.flatnonzero(result_bits != expected)
    assert observed_counts == expected_counts, {
        "counts": {
            f"0x{bits:08x}": (observed_counts[bits], expected_counts[bits])
            for bits in observed_counts.keys() | expected_counts.keys()
            if observed_counts[bits] != expected_counts[bits]
        },
        "positional_sample": [
            {
                "normalized_r": f"0x{int(source_bits[index]):08x}",
                "got": f"0x{int(result_bits[index]):08x}",
                "expected": f"0x{int(expected[index]):08x}",
            }
            for index in mismatches[:8]
        ],
    }
