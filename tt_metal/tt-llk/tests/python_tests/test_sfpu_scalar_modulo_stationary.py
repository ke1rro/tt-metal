# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blackhole gate for the test-only exponent-stationary magnitude reducer."""

import math
import struct
from dataclasses import dataclass

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
MAX_FINITE_BITS = 0x7F7FFFFF

DIVISOR_CASES = [
    (0x00800000, "smallest_normal"),
    (0x00800001, "low_mantissa_boundary"),
    (0x00800FFF, "split_boundary"),
    (0x00FFFFFF, "largest_eb_minus_126"),
    (0x40400000, "three"),
    (0x3DCCCCCD, "fp32_point_one"),
    (0x40FFFFFF, "next_down_eight"),
    (0x41000000, "eight"),
    (0x41000001, "next_up_eight"),
    (0x77800000, "power_two_eb_112"),
    (0x7F7FFFFF, "flt_max"),
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


def _scale_normal_bits(bits: int, exponent_shift: int) -> int:
    if bits == 0:
        return 0
    biased = (bits >> 23) & 0xFF
    shifted = biased + exponent_shift
    assert (
        0 < shifted < 0xFF
    ), f"non-normal exact scale: bits=0x{bits:08x} shift={exponent_shift}"
    return (bits & 0x807FFFFF) | (shifted << 23)


def _compose_exact_normalized_bits(significand: int) -> int:
    if significand == 0:
        return 0
    top_bit = significand.bit_length() - 1
    assert top_bit <= 23
    top_exponent = WORKING_EXPONENT - 23 + top_bit
    biased = top_exponent + 127
    normalized = significand << (23 - top_bit)
    return (biased << 23) | (normalized & MANTISSA_MASK)


def _exact_ratio_and_remainder(
    residual_bits: int, divisor_bits: int
) -> tuple[int, int]:
    if residual_bits < divisor_bits:
        return 0, residual_bits
    residual_m, residual_unit, _ = _normal_parts(residual_bits)
    divisor_m, divisor_unit, _ = _normal_parts(divisor_bits)
    shift = residual_unit - divisor_unit
    assert shift >= 0
    numerator = residual_m << shift
    quotient, remainder = divmod(numerator, divisor_m)
    return quotient, _compose_exact_normalized_bits(remainder)


def _exact_normalized_remainder_bits(input_bits: int, divisor_bits: int) -> int:
    magnitude_bits = input_bits & 0x7FFFFFFF
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
    rounded = math.floor(float(value) + 0.5)
    return min(max(rounded, 0), 0xFFFF)


def _stationary_oracle(input_bits: int, divisor_bits: int) -> tuple[int, list[int]]:
    magnitude_bits = input_bits & 0x7FFFFFFF
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
            quotient, residual_bits = _exact_ratio_and_remainder(
                residual_bits, normalized_divisor_bits
            )
            quotient_error = quotient_hat - quotient
            assert quotient_error in (0, 1), (
                f"q_hat contract failed input=0x{input_bits:08x} "
                f"divisor=0x{divisor_bits:08x} q={quotient} q_hat={quotient_hat}"
            )
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
            quotient, residual_bits = _exact_ratio_and_remainder(
                residual_bits, normalized_divisor_bits
            )
            assert quotient == 1

    expected_bits = _exact_normalized_remainder_bits(input_bits, divisor_bits)
    assert residual_bits == expected_bits
    return residual_bits, quotient_errors


def _floor_ratio(input_bits: int, divisor_bits: int) -> int:
    input_bits &= 0x7FFFFFFF
    if input_bits < divisor_bits:
        return 0
    input_m, input_unit, _ = _normal_parts(input_bits)
    divisor_m, divisor_unit, _ = _normal_parts(divisor_bits)
    return (input_m << (input_unit - divisor_unit)) // divisor_m


def _find_ratio_bits(divisor_bits: int, target: int) -> int | None:
    divisor = float(_from_bits(divisor_bits))
    product = divisor * target
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


def _dividend_bits(divisor_bits: int) -> list[int]:
    values = {0, divisor_bits}

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

    # Exercise both signs while keeping the reducer's output magnitude-only.
    positive = sorted(values)
    signed = positive + [bits | 0x80000000 for bits in positive if bits]
    return signed


@dataclass
class StationaryConstants(TemplateParameter):
    divisor_bits: int

    def convert_to_cpp(self) -> str:
        normalized, reciprocal, start, initial, high = _constants(self.divisor_bits)
        return "\n".join(
            (
                f"constexpr std::uint32_t SCALAR_NORMALIZED_DIVISOR = {normalized}u;",
                f"constexpr std::uint32_t SCALAR_RECIPROCAL_UP = {reciprocal}u;",
                f"constexpr int SCALAR_START_SHIFT = {start};",
                f"constexpr int SCALAR_INITIAL_SHIFT = {initial};",
                f"constexpr unsigned SCALAR_HIGH_MANTISSA = {high}u;",
            )
        )


def _run(divisor_bits: int) -> tuple[np.ndarray, np.ndarray]:
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("the first stationary reducer device gate is Blackhole-only")

    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    input_dimensions = [32, 32]
    values = np.asarray(_dividend_bits(divisor_bits), dtype=np.uint32).view(np.float32)
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
        "sources/sfpu_scalar_modulo_stationary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            StationaryConstants(divisor_bits),
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
    result = np.asarray(configuration.run().result, dtype=np.float32)
    return result, np.asarray(src_a, dtype=np.float32)


@pytest.mark.parametrize(
    ("divisor_bits", "case_name"),
    DIVISOR_CASES,
    ids=[case_name for _, case_name in DIVISOR_CASES],
)
def test_scalar_modulo_stationary_normalized_bits(divisor_bits, case_name):
    del case_name
    result, source = _run(divisor_bits)
    expected = np.empty(source.size, dtype=np.uint32)
    observed_quotient_errors = set()
    for index, input_bits in enumerate(source.view(np.uint32).reshape(-1)):
        expected[index], errors = _stationary_oracle(int(input_bits), divisor_bits)
        observed_quotient_errors.update(errors)

    result_bits = result.view(np.uint32).reshape(-1)
    mismatches = np.flatnonzero(result_bits != expected)
    assert mismatches.size == 0, [
        {
            "input": f"0x{int(source.view(np.uint32).reshape(-1)[index]):08x}",
            "got": f"0x{int(result_bits[index]):08x}",
            "expected": f"0x{int(expected[index]):08x}",
        }
        for index in mismatches[:16]
    ]
    assert observed_quotient_errors <= {0, 1}
