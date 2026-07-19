# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact raw-bit Blackhole gate for the explicit FastBounded contract."""

import math
from collections import Counter
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
from test_sfpu_scalar_modulo_stationary_combined import (
    MANTISSA_MASK,
    MAX_FINITE_BITS,
    SIGN_MASK,
    ScalarModuloKind,
    _bits,
    _dividend_magnitudes,
    _exact_normalized_remainder_bits,
    _expected_raw_bits,
    _find_ratio_bits,
    _floor_ratio,
    _from_bits,
    _normal_parts,
    _scale_normal_bits,
)

WORKING_EXPONENT = 103
FINAL_EXPONENT = 111
STAGE_SPLIT = FINAL_EXPONENT - WORKING_EXPONENT
QUOTIENT_BOUND_EXPONENT = 22


DIVISORS = [
    (0x40400000, "three"),
    (0x00800000, "smallest_normal"),
    (0x00800001, "smallest_normal_next"),
    (0x0C000000, "eb_minus_103"),
    (0x0B800000, "eb_minus_104"),
    (0x77000000, "eb_111"),
    (0x77800000, "eb_112"),
    (0x7F7FFFFF, "flt_max"),
    (0x4046BEAE, "direct_product_regression"),
    (0x40687B64, "component_regression"),
    (0x537F6BDB, "underestimate_regression"),
]

REGRESSION_INPUTS = {
    0x4046BEAE: (0x4B46B792,),
    0x40687B64: (0x4B5DF984,),
    0x537F6BDB: (0x5E6E2232,),
}


@dataclass(frozen=True)
class FastBoundedCase:
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


FAST_BOUNDED_CASES = [
    FastBoundedCase(
        divisor_bits, divisor_name, kind, dividend_negative, divisor_negative
    )
    for divisor_bits, divisor_name in DIVISORS
    for kind in ScalarModuloKind
    for dividend_negative in (False, True)
    for divisor_negative in (False, True)
]


def _working_constants(divisor_bits: int) -> tuple[int, int, int, int, int]:
    _, _, divisor_exponent = _normal_parts(divisor_bits)
    normalized_divisor = _scale_normal_bits(
        divisor_bits, FINAL_EXPONENT - divisor_exponent
    )
    working_divisor = _scale_normal_bits(
        divisor_bits, WORKING_EXPONENT - divisor_exponent
    )
    reciprocal = np.float32(np.float32(1.0) / _from_bits(working_divisor))
    reciprocal_bits = _bits(reciprocal)
    reciprocal_bias = 0 if (divisor_bits & MANTISSA_MASK) == 0 else 2

    divisor_high_bits = normalized_divisor & ~0xFFF
    high_mantissa = (divisor_high_bits & MANTISSA_MASK) >> 11
    assert (high_mantissa & 1) == 0
    return (
        normalized_divisor,
        working_divisor,
        reciprocal_bits,
        high_mantissa,
        reciprocal_bias,
    )


def _working_input_bits(input_bits: int, divisor_bits: int) -> int:
    magnitude_bits = input_bits & ~SIGN_MASK
    _, _, divisor_exponent = _normal_parts(divisor_bits)
    return _scale_normal_bits(magnitude_bits, WORKING_EXPONENT - divisor_exponent)


def _contract_state(input_bits: int, divisor_bits: int) -> tuple[bool, int, int, int]:
    magnitude_bits = input_bits & ~SIGN_MASK
    if magnitude_bits == 0 or magnitude_bits < divisor_bits:
        return True, 0, 0, 0

    _, _, divisor_exponent = _normal_parts(divisor_bits)
    _, _, input_exponent = _normal_parts(magnitude_bits)
    working_input_exponent = input_exponent + WORKING_EXPONENT - divisor_exponent
    if working_input_exponent > 127:
        return False, 0, 0, 0x7F800000

    _, _, reciprocal_bits, _, _ = _working_constants(divisor_bits)
    working_input_bits = _working_input_bits(magnitude_bits, divisor_bits)
    scaled = np.float32(_from_bits(working_input_bits) * _from_bits(reciprocal_bits))
    scaled_bits = _bits(scaled)
    if not math.isfinite(float(scaled)):
        return False, 0, 0, scaled_bits
    assert scaled >= np.float32(0.0)
    safe = scaled == np.float32(0.0) or _normal_parts(scaled_bits)[2] < 22
    quotient_exact = _floor_ratio(magnitude_bits, divisor_bits)
    quotient_hat = math.floor(float(scaled))
    return safe, quotient_hat - quotient_exact, quotient_hat, scaled_bits


def _precision_span(value: int) -> int:
    if value == 0:
        return 0
    magnitude = abs(value)
    trailing = (magnitude & -magnitude).bit_length() - 1
    return magnitude.bit_length() - trailing


def _significand_components(significand: int, component_bits: int) -> list[int]:
    components = []
    remaining = 24
    while remaining:
        width = min(component_bits, remaining)
        remaining -= width
        chunk = (significand >> remaining) & ((1 << width) - 1)
        if chunk:
            components.append(chunk << remaining)
    return components


def _compose_normal_bits(significand: int, unit_exponent: int) -> int:
    if significand == 0:
        return 0
    top_bit = significand.bit_length() - 1
    assert _precision_span(significand) <= 24
    exponent = unit_exponent + top_bit
    assert -126 <= exponent <= 127
    shift = top_bit - 23
    if shift > 0:
        assert (significand & ((1 << shift) - 1)) == 0
        normalized = significand >> shift
    else:
        normalized = significand << -shift
    return ((exponent + 127) << 23) | (normalized & MANTISSA_MASK)


def _reciprocal_up_bits(reciprocal_bits: int, ulp_bias: int) -> int:
    return reciprocal_bits + ulp_bias


def _audit_stage(
    residual_integer: int,
    divisor_significand: int,
    divisor_scale: int,
    reciprocal_bits: int,
) -> tuple[int, int]:
    divisor_integer = divisor_significand << divisor_scale
    if residual_integer < divisor_integer:
        return residual_integer, 0

    unit_exponent = WORKING_EXPONENT - 23
    residual_bits = _compose_normal_bits(residual_integer, unit_exponent)
    quotient_product = np.float32(
        _from_bits(residual_bits) * _from_bits(reciprocal_bits)
    )
    quotient_hat = math.floor(float(quotient_product))
    quotient_exact = residual_integer // divisor_integer
    quotient_error = quotient_hat - quotient_exact
    assert quotient_error in (0, 1)

    for component in _significand_components(divisor_significand, 12):
        component <<= divisor_scale
        product = quotient_hat * component
        assert _precision_span(product) <= 28
        residual_integer -= product
        assert _precision_span(residual_integer) <= 24

    if quotient_error:
        residual_integer += divisor_integer
    assert 0 <= residual_integer < divisor_integer
    return residual_integer, quotient_error


def _audit_two_stage_reducer(input_bits: int, divisor_bits: int) -> list[int]:
    magnitude_bits = input_bits & ~SIGN_MASK
    safe, global_error, _, _ = _contract_state(magnitude_bits, divisor_bits)
    assert safe
    assert global_error in (-1, 0, 1)
    if magnitude_bits < divisor_bits:
        return []

    divisor_m, divisor_unit, divisor_exponent = _normal_parts(divisor_bits)
    input_m, input_unit, _ = _normal_parts(magnitude_bits)
    exponent_delta = input_unit - divisor_unit
    assert exponent_delta >= 0
    residual_integer = input_m << exponent_delta

    _, _, reciprocal_bits, _, reciprocal_bias = _working_constants(divisor_bits)
    reciprocal_up = _reciprocal_up_bits(reciprocal_bits, reciprocal_bias)
    stage_one_reciprocal = _scale_normal_bits(reciprocal_up, -STAGE_SPLIT)

    errors = []
    residual_integer, error = _audit_stage(
        residual_integer,
        divisor_m,
        STAGE_SPLIT,
        stage_one_reciprocal,
    )
    errors.append(error)
    residual_integer, error = _audit_stage(
        residual_integer, divisor_m, 0, reciprocal_up
    )
    errors.append(error)

    exact_remainder = (input_m << exponent_delta) % divisor_m
    assert residual_integer == exact_remainder
    normalized_bits = _compose_normal_bits(
        residual_integer, WORKING_EXPONENT - 23 + STAGE_SPLIT
    )
    assert normalized_bits == _exact_normalized_remainder_bits(
        magnitude_bits, divisor_bits
    )
    return errors


def _last_safe_boundary(divisor_bits: int) -> tuple[int, int] | None:
    if _contract_state(MAX_FINITE_BITS, divisor_bits)[0]:
        return None

    low = divisor_bits
    high = MAX_FINITE_BITS
    while low + 1 < high:
        middle = (low + high) // 2
        if _contract_state(middle, divisor_bits)[0]:
            low = middle
        else:
            high = middle
    assert _contract_state(low, divisor_bits)[0]
    assert not _contract_state(high, divisor_bits)[0]
    return low, high


def _fast_bounded_magnitudes(divisor_bits: int) -> list[int]:
    values = set(_dividend_magnitudes(divisor_bits))
    for quotient in (
        0,
        1,
        2,
        (1 << 16) - 1,
        (1 << 21) - 1,
        1 << 21,
        (1 << 21) + 1,
        (1 << 22) - 2,
        (1 << 22) - 1,
        1 << 22,
    ):
        candidate = _find_ratio_bits(divisor_bits, quotient)
        if candidate is None:
            continue
        for delta in (-1, 0, 1):
            adjacent = candidate + delta
            if 0x00800000 <= adjacent <= MAX_FINITE_BITS:
                values.add(adjacent)

    boundary = _last_safe_boundary(divisor_bits)
    if boundary is not None:
        last_safe, first_unsafe = boundary
        values.update(
            bits
            for bits in (last_safe - 1, last_safe, first_unsafe, first_unsafe + 1)
            if 0x00800000 <= bits <= MAX_FINITE_BITS
        )
    values.update(REGRESSION_INPUTS.get(divisor_bits, ()))
    return sorted(bits for bits in values if _contract_state(bits, divisor_bits)[0])


def _input_bits(case: FastBoundedCase) -> list[int]:
    return [
        bits | (SIGN_MASK if case.dividend_negative else 0)
        for bits in _fast_bounded_magnitudes(case.divisor_bits)
    ]


@dataclass
class FastBoundedConstants(TemplateParameter):
    case: FastBoundedCase

    def convert_to_cpp(self) -> str:
        normalized, _, reciprocal, high_mantissa, reciprocal_bias = _working_constants(
            self.case.divisor_bits
        )
        _, _, divisor_exponent = _normal_parts(self.case.divisor_bits)
        return "\n".join(
            (
                f"constexpr std::uint32_t SCALAR_PHYSICAL_DIVISOR = {self.case.divisor_bits}u;",
                f"constexpr std::uint32_t SCALAR_NORMALIZED_DIVISOR = {normalized}u;",
                f"constexpr std::uint32_t FAST_WORKING_RECIPROCAL = {reciprocal}u;",
                f"constexpr int SCALAR_DIVISOR_EXPONENT = {divisor_exponent};",
                f"constexpr int FAST_INPUT_EXPONENT_SHIFT = {WORKING_EXPONENT - divisor_exponent};",
                f"constexpr unsigned FAST_HIGH_MANTISSA = {high_mantissa}u;",
                f"constexpr unsigned FAST_RECIPROCAL_ULP_BIAS = {reciprocal_bias}u;",
                f"constexpr int FAST_BOUNDED_KIND = {int(self.case.kind)};",
                f"constexpr bool FAST_BOUNDED_DIVIDEND_NEGATIVE = {str(self.case.dividend_negative).lower()};",
                f"constexpr bool FAST_BOUNDED_DIVISOR_NEGATIVE = {str(self.case.divisor_negative).lower()};",
            )
        )


def _run(case: FastBoundedCase) -> tuple[np.ndarray, np.ndarray]:
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("FastBoundedExactBH is Blackhole-only")

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
        "sources/sfpu_scalar_modulo_fast_bounded_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            FastBoundedConstants(case),
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


def test_fast_bounded_contract_and_quotient_audit():
    global_errors = set()
    local_errors = set()
    output_magnitudes = set()
    observed_boundary = False
    for divisor_bits, _ in DIVISORS:
        boundary = _last_safe_boundary(divisor_bits)
        if boundary is not None:
            observed_boundary = True
            assert _contract_state(boundary[0], divisor_bits)[0]
            assert not _contract_state(boundary[1], divisor_bits)[0]
        for input_bits in _fast_bounded_magnitudes(divisor_bits):
            safe, error, _, _ = _contract_state(input_bits, divisor_bits)
            assert safe
            assert error in (-1, 0, 1)
            global_errors.add(error)
            local_errors.update(_audit_two_stage_reducer(input_bits, divisor_bits))
        for case in FAST_BOUNDED_CASES:
            if case.divisor_bits != divisor_bits:
                continue
            output_magnitudes.update(
                _expected_raw_bits(input_bits, case) & ~SIGN_MASK
                for input_bits in _input_bits(case)
            )
    assert observed_boundary
    assert global_errors == {-1, 0, 1}
    assert local_errors == {0, 1}
    assert 0 in output_magnitudes
    assert 1 in output_magnitudes
    assert 0x007FFFFF in output_magnitudes
    assert any(0 < bits < 0x00800000 for bits in output_magnitudes)
    assert any(0x00800000 <= bits <= MAX_FINITE_BITS for bits in output_magnitudes)


@pytest.mark.parametrize("case", FAST_BOUNDED_CASES, ids=lambda case: case.name)
def test_scalar_modulo_fast_bounded_raw_bits(case):
    result, source = _run(case)
    source_bits = source.view(np.uint32).reshape(-1)
    expected = np.asarray(
        [_expected_raw_bits(int(bits), case) for bits in source_bits],
        dtype=np.uint32,
    )

    for bits in source_bits:
        safe, global_error, _, _ = _contract_state(int(bits), case.divisor_bits)
        assert safe
        assert global_error in (-1, 0, 1)
        assert set(_audit_two_stage_reducer(int(bits), case.divisor_bits)) <= {0, 1}

    result_bits = result.reshape(-1)
    observed_counts = Counter(int(bits) for bits in result_bits)
    expected_counts = Counter(int(bits) for bits in expected)
    assert observed_counts == expected_counts, {
        f"0x{bits:08x}": (observed_counts[bits], expected_counts[bits])
        for bits in observed_counts.keys() | expected_counts.keys()
        if observed_counts[bits] != expected_counts[bits]
    }
