# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end raw-bit Blackhole gate for stationary scalar modulo."""

import math
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
CHUNK_STEP = 15
MANTISSA_MASK = 0x007FFFFF
SIGN_MASK = 0x80000000
MAX_FINITE_BITS = 0x7F7FFFFF


class ScalarModuloKind(IntEnum):
    FMOD = 0
    FLOOR_REMAINDER = 1


DIVISORS = [
    (0x40400000, "three"),
    (0x00800000, "smallest_normal"),
    (0x00800001, "smallest_normal_next"),
    (0x0C000000, "eb_minus_103"),
    (0x0B800000, "eb_minus_104"),
    (0x77000000, "eb_111"),
    (0x77800000, "eb_112"),
    (0x7F7FFFFF, "flt_max"),
]


@dataclass(frozen=True)
class CombinedCase:
    divisor_bits: int
    divisor_name: str
    kind: ScalarModuloKind
    dividend_negative: bool
    divisor_negative: bool

    @property
    def name(self) -> str:
        operation = "fmod" if self.kind == ScalarModuloKind.FMOD else "floor"
        dividend = "aneg" if self.dividend_negative else "apos"
        divisor = "bneg" if self.divisor_negative else "bpos"
        return f"{self.divisor_name}_{operation}_{dividend}_{divisor}"


COMBINED_CASES = [
    CombinedCase(divisor_bits, divisor_name, kind, dividend_negative, divisor_negative)
    for divisor_bits, divisor_name in DIVISORS
    for kind in ScalarModuloKind
    for dividend_negative in (False, True)
    for divisor_negative in (False, True)
]


def _bits(value: float | np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _from_bits(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


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


def _exact_remainder_lattice(magnitude_bits: int, divisor_bits: int) -> tuple[int, int]:
    if magnitude_bits == 0:
        return 0, -149

    input_m, input_unit, _ = _normal_parts(magnitude_bits)
    if magnitude_bits < divisor_bits:
        return input_m, input_unit

    divisor_m, divisor_unit, _ = _normal_parts(divisor_bits)
    shift = input_unit - divisor_unit
    assert shift >= 0
    return (input_m << shift) % divisor_m, divisor_unit


def _exact_difference_from_divisor(
    remainder_m: int, remainder_unit: int, divisor_bits: int
) -> tuple[int, int]:
    divisor_m, divisor_unit, _ = _normal_parts(divisor_bits)
    common_unit = min(remainder_unit, divisor_unit)
    divisor_integer = divisor_m << (divisor_unit - common_unit)
    remainder_integer = remainder_m << (remainder_unit - common_unit)
    assert divisor_integer > remainder_integer
    return divisor_integer - remainder_integer, common_unit


def _expected_raw_bits(input_bits: int, case: CombinedCase) -> int:
    magnitude_bits = input_bits & ~SIGN_MASK
    remainder_m, remainder_unit = _exact_remainder_lattice(
        magnitude_bits, case.divisor_bits
    )
    signs_differ = case.dividend_negative != case.divisor_negative

    result_m, result_unit = remainder_m, remainder_unit
    if (
        case.kind == ScalarModuloKind.FLOOR_REMAINDER
        and signs_differ
        and remainder_m != 0
    ):
        result_m, result_unit = _exact_difference_from_divisor(
            remainder_m, remainder_unit, case.divisor_bits
        )

    magnitude_result = _compose_rne_bits(result_m, result_unit)
    if remainder_m == 0 or case.kind == ScalarModuloKind.FMOD:
        negative = case.dividend_negative
    else:
        negative = case.divisor_negative
    return magnitude_result | (SIGN_MASK if negative else 0)


def _scale_normal_bits(bits: int, exponent_shift: int) -> int:
    if bits == 0:
        return 0
    biased = (bits >> 23) & 0xFF
    shifted = biased + exponent_shift
    assert 0 < shifted < 0xFF
    return (bits & 0x807FFFFF) | (shifted << 23)


def _compose_exact_normalized_bits(significand: int) -> int:
    if significand == 0:
        return 0
    top_bit = significand.bit_length() - 1
    assert top_bit <= 23
    top_exponent = WORKING_EXPONENT - 23 + top_bit
    normalized = significand << (23 - top_bit)
    return ((top_exponent + 127) << 23) | (normalized & MANTISSA_MASK)


def _exact_ratio_and_normalized_remainder(
    residual_bits: int, divisor_bits: int
) -> tuple[int, int]:
    if residual_bits < divisor_bits:
        return 0, residual_bits
    residual_m, residual_unit, _ = _normal_parts(residual_bits)
    divisor_m, divisor_unit, _ = _normal_parts(divisor_bits)
    shift = residual_unit - divisor_unit
    assert shift >= 0
    quotient, remainder = divmod(residual_m << shift, divisor_m)
    return quotient, _compose_exact_normalized_bits(remainder)


def _exact_normalized_remainder_bits(input_bits: int, divisor_bits: int) -> int:
    magnitude_bits = input_bits & ~SIGN_MASK
    if magnitude_bits == 0:
        return 0
    _, _, divisor_exponent = _normal_parts(divisor_bits)
    normalization_shift = WORKING_EXPONENT - divisor_exponent
    if magnitude_bits < divisor_bits:
        return _scale_normal_bits(magnitude_bits, normalization_shift)

    input_m, input_unit, _ = _normal_parts(magnitude_bits)
    divisor_m, divisor_unit, _ = _normal_parts(divisor_bits)
    exponent_delta = input_unit - divisor_unit
    assert exponent_delta >= 0
    remainder = (input_m * pow(2, exponent_delta, divisor_m)) % divisor_m
    return _compose_exact_normalized_bits(remainder)


def _reciprocal_up_bits(normalized_divisor_bits: int) -> int:
    normalized_divisor = _from_bits(normalized_divisor_bits)
    reciprocal = np.float32(np.float32(1.0) / normalized_divisor)
    if normalized_divisor_bits & MANTISSA_MASK:
        reciprocal = np.nextafter(reciprocal, np.float32(np.inf), dtype=np.float32)
        reciprocal = np.nextafter(reciprocal, np.float32(np.inf), dtype=np.float32)
    return _bits(reciprocal)


def _constants(divisor_bits: int) -> tuple[int, int, int, int, int]:
    _, _, divisor_exponent = _normal_parts(divisor_bits)
    normalization_shift = WORKING_EXPONENT - divisor_exponent
    normalized_divisor_bits = _scale_normal_bits(divisor_bits, normalization_shift)
    reciprocal_bits = _reciprocal_up_bits(normalized_divisor_bits)
    start_shift = max(normalization_shift, 0)
    initial_shift = min(normalization_shift, 0)
    divisor_high_bits = normalized_divisor_bits & ~0xFFF
    high_mantissa = (divisor_high_bits & MANTISSA_MASK) >> 11
    assert (high_mantissa & 1) == 0
    return (
        normalized_divisor_bits,
        reciprocal_bits,
        start_shift,
        initial_shift,
        high_mantissa,
    )


def _nearest_away_uint16(value: np.float32) -> int:
    return min(max(math.floor(float(value) + 0.5), 0), 0xFFFF)


def _stationary_quotient_errors(input_bits: int, divisor_bits: int) -> list[int]:
    magnitude_bits = input_bits & ~SIGN_MASK
    normalized_divisor_bits, reciprocal_bits, shift, initial_shift, _ = _constants(
        divisor_bits
    )
    prehalved = magnitude_bits != 0 and _normal_parts(magnitude_bits)[2] == 127
    residual_bits = (
        _scale_normal_bits(magnitude_bits, -1) if prehalved else magnitude_bits
    )
    if initial_shift and residual_bits:
        residual_bits = _scale_normal_bits(residual_bits, initial_shift)

    quotient_errors = []
    while True:
        if residual_bits >= normalized_divisor_bits:
            scaled_quotient = np.float32(
                _from_bits(residual_bits) * _from_bits(reciprocal_bits)
            )
            quotient_hat = _nearest_away_uint16(scaled_quotient)
            quotient, residual_bits = _exact_ratio_and_normalized_remainder(
                residual_bits, normalized_divisor_bits
            )
            quotient_error = quotient_hat - quotient
            assert quotient_error in (0, 1)
            quotient_errors.append(quotient_error)

        if shift == 0:
            break
        decrement = min(CHUNK_STEP, shift)
        shift -= decrement
        if residual_bits:
            residual_bits = _scale_normal_bits(residual_bits, decrement)

    if prehalved and residual_bits:
        residual_bits = _scale_normal_bits(residual_bits, 1)
        if residual_bits >= normalized_divisor_bits:
            quotient, residual_bits = _exact_ratio_and_normalized_remainder(
                residual_bits, normalized_divisor_bits
            )
            assert quotient == 1

    assert residual_bits == _exact_normalized_remainder_bits(input_bits, divisor_bits)
    return quotient_errors


def _floor_ratio(input_bits: int, divisor_bits: int) -> int:
    if input_bits < divisor_bits:
        return 0
    input_m, input_unit, _ = _normal_parts(input_bits)
    divisor_m, divisor_unit, _ = _normal_parts(divisor_bits)
    return (input_m << (input_unit - divisor_unit)) // divisor_m


def _find_ratio_bits(divisor_bits: int, target: int) -> int | None:
    product = float(_from_bits(divisor_bits)) * target
    if not math.isfinite(product) or product > float(np.finfo(np.float32).max):
        return None
    candidate = _bits(np.float32(product))
    for _ in range(8):
        quotient = _floor_ratio(candidate, divisor_bits)
        if quotient == target:
            return candidate
        candidate += 1 if quotient < target else -1
        if not (0x00800000 <= candidate <= MAX_FINITE_BITS):
            return None
    return None


def _dividend_magnitudes(divisor_bits: int) -> list[int]:
    values = {0, 0x00800000, 0x3F800000, divisor_bits}
    if divisor_bits > 0x00800000:
        values.add(divisor_bits - 1)
    if divisor_bits < MAX_FINITE_BITS:
        values.add(divisor_bits + 1)

    _, _, divisor_exponent = _normal_parts(divisor_bits)
    if divisor_exponent < 127:
        exact_multiple = _scale_normal_bits(divisor_bits, 1)
        values.add(exact_multiple)
        if exact_multiple < MAX_FINITE_BITS:
            values.add(exact_multiple + 1)

    for quotient in (65534, 65535):
        candidate = _find_ratio_bits(divisor_bits, quotient)
        if candidate is not None:
            values.add(candidate)

    boundary = _find_ratio_bits(divisor_bits, 65536)
    if boundary is not None and boundary > 0x00800000:
        values.add(boundary - 1)

    values.update(
        {
            0x7EFFFFFE,
            0x7EFFFFFF,
            0x7F000000,
            0x7F000001,
            0x7F7FFFFE,
            0x7F7FFFFF,
        }
    )
    return sorted(values)


def _input_bits(case: CombinedCase) -> list[int]:
    return [
        bits | (SIGN_MASK if case.dividend_negative else 0)
        for bits in _dividend_magnitudes(case.divisor_bits)
    ]


@dataclass
class CombinedConstants(TemplateParameter):
    case: CombinedCase

    def convert_to_cpp(self) -> str:
        normalized, reciprocal, start, initial, high = _constants(
            self.case.divisor_bits
        )
        _, _, divisor_exponent = _normal_parts(self.case.divisor_bits)
        return "\n".join(
            (
                f"constexpr std::uint32_t SCALAR_PHYSICAL_DIVISOR = {self.case.divisor_bits}u;",
                f"constexpr std::uint32_t SCALAR_NORMALIZED_DIVISOR = {normalized}u;",
                f"constexpr std::uint32_t SCALAR_RECIPROCAL_UP = {reciprocal}u;",
                f"constexpr int SCALAR_DIVISOR_EXPONENT = {divisor_exponent};",
                f"constexpr int SCALAR_START_SHIFT = {start};",
                f"constexpr int SCALAR_INITIAL_SHIFT = {initial};",
                f"constexpr unsigned SCALAR_HIGH_MANTISSA = {high}u;",
                f"constexpr int COMBINED_KIND = {int(self.case.kind)};",
                f"constexpr bool COMBINED_DIVIDEND_NEGATIVE = {str(self.case.dividend_negative).lower()};",
                f"constexpr bool COMBINED_DIVISOR_NEGATIVE = {str(self.case.divisor_negative).lower()};",
            )
        )


def _run(case: CombinedCase) -> tuple[np.ndarray, np.ndarray]:
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("the exact combined stationary kernel is Blackhole-only")

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
        "sources/sfpu_scalar_modulo_stationary_combined_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            CombinedConstants(case),
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


@pytest.mark.parametrize("case", COMBINED_CASES, ids=lambda case: case.name)
def test_scalar_modulo_stationary_combined_raw_bits(case):
    result, source = _run(case)
    source_bits = source.view(np.uint32).reshape(-1)
    expected = np.asarray(
        [_expected_raw_bits(int(bits), case) for bits in source_bits],
        dtype=np.uint32,
    )

    observed_quotient_errors = set()
    _, _, divisor_exponent = _normal_parts(case.divisor_bits)
    for bits in source_bits:
        magnitude_bits = int(bits) & ~SIGN_MASK
        high_divisor_bypass = (
            divisor_exponent > WORKING_EXPONENT and magnitude_bits < case.divisor_bits
        )
        if not high_divisor_bypass:
            observed_quotient_errors.update(
                _stationary_quotient_errors(int(bits), case.divisor_bits)
            )
    assert observed_quotient_errors <= {0, 1}

    result_bits = result.reshape(-1)
    observed_counts = Counter(int(bits) for bits in result_bits)
    expected_counts = Counter(int(bits) for bits in expected)
    assert observed_counts == expected_counts, {
        f"0x{bits:08x}": (observed_counts[bits], expected_counts[bits])
        for bits in observed_counts.keys() | expected_counts.keys()
        if observed_counts[bits] != expected_counts[bits]
    }
