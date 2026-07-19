# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Selected raw-FP32 checks for the test-only robust fixed schedule."""

import collections
import struct
from dataclasses import dataclass

import numpy as np
import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)

COUNTEREXAMPLES = {
    3.0: 50331656.0,
    5.0: 83886104.0,
    7.0: 116040560.0,
    10.0: 167772208.0,
    float(np.float32(0.1)): 1677722.125,
    float(np.float32(0.3)): 5033166.5,
    8.0: 134217736.0,
}


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _is_power_of_two(divisor_bits: int) -> bool:
    return (divisor_bits & 0x7FFFFF) == 0


def _normal_parts(bits: int) -> tuple[int, int]:
    biased = (bits >> 23) & 0xFF
    assert 0 < biased < 0xFF
    return (1 << 23) | (bits & 0x7FFFFF), biased - 127 - 23


def _round_shift_right_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    truncated = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return truncated + int(remainder > halfway or (remainder == halfway and (truncated & 1)))


def _compose_rne_bits(significand: int, unit_exponent: int) -> int:
    if significand == 0:
        return 0
    top = unit_exponent + significand.bit_length() - 1
    if top < -126:
        subnormal = _round_shift_right_even(significand, -149 - unit_exponent)
        return subnormal

    normalized = _round_shift_right_even(significand, significand.bit_length() - 24)
    if normalized == 1 << 24:
        normalized >>= 1
        top += 1
    assert top <= 127
    return ((top + 127) << 23) | (normalized & 0x7FFFFF)


def _exact_magnitude_mod_bits(input_bits: int, divisor_bits: int) -> int:
    if input_bits == 0:
        return 0
    if input_bits < divisor_bits:
        return input_bits
    input_m, input_unit = _normal_parts(input_bits)
    divisor_m, divisor_unit = _normal_parts(divisor_bits)
    delta = input_unit - divisor_unit
    assert delta >= 0
    remainder = (input_m << delta) % divisor_m
    return _compose_rne_bits(remainder, divisor_unit)


def _exact_expected_bits(src: np.ndarray, scalar: float, mathop: MathOperation) -> np.ndarray:
    scalar_bits = _bits(float(np.float32(scalar)))
    scalar_sign = scalar_bits & 0x80000000
    divisor_bits = scalar_bits & 0x7FFFFFFF
    output = np.empty(src.size, dtype=np.uint32)

    for index, signed_input_bits in enumerate(src.view(np.uint32).reshape(-1)):
        input_bits = int(signed_input_bits)
        input_sign = input_bits & 0x80000000
        magnitude_bits = input_bits & 0x7FFFFFFF
        remainder_bits = _exact_magnitude_mod_bits(magnitude_bits, divisor_bits)

        if mathop == MathOperation.Fmod:
            output[index] = remainder_bits | input_sign
        elif remainder_bits == 0:
            output[index] = input_sign
        elif input_sign != scalar_sign:
            remainder_m, remainder_unit = _normal_parts(remainder_bits)
            divisor_m, divisor_unit = _normal_parts(divisor_bits)
            common_unit = min(remainder_unit, divisor_unit)
            aligned_remainder = remainder_m << (remainder_unit - common_unit)
            aligned_divisor = divisor_m << (divisor_unit - common_unit)
            output[index] = _compose_rne_bits(aligned_divisor - aligned_remainder, common_unit) | scalar_sign
        else:
            output[index] = remainder_bits | scalar_sign
    return output.reshape(src.shape)


def _constants(scalar: float) -> tuple[int, int, int, int]:
    scalar32 = np.float32(scalar)
    divisor = np.float32(abs(scalar32))
    divisor_bits = _bits(float(divisor))
    divisor_exponent = ((divisor_bits >> 23) & 0xFF) - 127
    start_shift = max(112 - divisor_exponent, 0)

    reciprocal = np.float32(np.float32(1.0) / divisor)
    if not _is_power_of_two(divisor_bits):
        reciprocal = np.nextafter(reciprocal, np.float32(np.inf), dtype=np.float32)
        reciprocal = np.nextafter(reciprocal, np.float32(np.inf), dtype=np.float32)
    reciprocal_bits = _bits(float(reciprocal))
    reciprocal_biased = (reciprocal_bits >> 23) & 0xFF
    initial_biased = reciprocal_biased - start_shift
    assert 0 < initial_biased < 0xFF
    initial_inverse_bits = (reciprocal_bits & 0x807FFFFF) | (initial_biased << 23)

    # SFPSETMAN writes immediate<<11.  Keeping the low immediate bit clear
    # splits the divisor into two <=12-significand-bit components.
    divisor_high_bits = divisor_bits & ~0xFFF
    high_mantissa = (divisor_high_bits & 0x7FFFFF) >> 11
    assert (high_mantissa & 1) == 0
    return _bits(float(scalar32)), initial_inverse_bits, start_shift, high_mantissa


def _seeded_normal_values(divisor: float, count: int = 231) -> list[float]:
    """Normal inputs below the documented top-exponent overflow exclusion."""

    rng = np.random.default_rng(_bits(divisor) ^ 0x5F17A11)
    bits = rng.integers(0x00800000, 0x7E000000, size=count, dtype=np.uint32)
    bits[1::2] |= np.uint32(0x80000000)
    return bits.view(np.float32).astype(float).tolist()


def _max_exponent_values() -> list[float]:
    bits = np.asarray([0x7F7FFF9D, 0x7F7FFFBC, 0x7F7FFFFE, 0x7F7FFFFF], dtype=np.uint32)
    values = bits.view(np.float32)
    return values.astype(float).tolist() + (-values).astype(float).tolist()


def _values_from_bits(bits: list[int]) -> list[float]:
    return np.asarray(bits, dtype=np.uint32).view(np.float32).astype(float).tolist()


@dataclass
class FixedScheduleConstants(TemplateParameter):
    scalar: float

    def convert_to_cpp(self) -> str:
        divisor, initial_inverse, start_shift, high_mantissa = _constants(self.scalar)
        return "\n".join(
            (
                f"constexpr std::uint32_t SCALAR_DIVISOR = {divisor}u;",
                f"constexpr std::uint32_t SCALAR_INITIAL_INVERSE = {initial_inverse}u;",
                f"constexpr int SCALAR_START_SHIFT = {start_shift};",
                f"constexpr unsigned SCALAR_HIGH_MANTISSA = {high_mantissa}u;",
            )
        )


def _run(values: list[float], mathop: MathOperation, scalar: float, diagnostic: bool = False):
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    input_dimensions = [32, 32]
    dest_acc = DestAccumulation.Yes

    src_a, tile_count_a, src_b, tile_count_b = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.custom(values),
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/sfpu_scalar_modulo_fixed_schedule_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            FAST_MODE(FastMode.Yes if diagnostic else FastMode.No),
            MATH_OP(mathop=mathop),
            FixedScheduleConstants(scalar),
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
        dest_acc=dest_acc,
        unpack_to_dest=True,
    )
    return np.asarray(configuration.run().result, dtype=np.float32), np.asarray(src_a, dtype=np.float32)


@pytest.mark.parametrize("mathop", [MathOperation.Fmod, MathOperation.Remainder])
@pytest.mark.parametrize("scalar_sign", [1.0, -1.0])
@pytest.mark.parametrize("divisor", list(COUNTEREXAMPLES))
def test_scalar_modulo_fixed_schedule_raw_bits(mathop, scalar_sign, divisor):
    counterexample = COUNTEREXAMPLES[divisor]
    center = np.float32(np.float32(2.0**24) * np.float32(divisor))
    values = [
        0.0,
        -0.0,
        1.0,
        divisor - 1.0,
        divisor,
        divisor + 1.0,
        -1.0,
        -(divisor - 1.0),
        -divisor,
        -(divisor + 1.0),
        counterexample,
        -counterexample,
        float(np.nextafter(center, np.float32(-np.inf), dtype=np.float32)),
        float(center),
        float(np.nextafter(center, np.float32(np.inf), dtype=np.float32)),
    ]
    if divisor == 3.0:
        values.extend((50855936.0, -50855936.0))
    values.extend(_max_exponent_values())
    values.extend(_seeded_normal_values(divisor))

    scalar = scalar_sign * divisor
    result, src = _run(values, mathop, scalar)
    expected_bits = _exact_expected_bits(src, scalar, mathop)
    assert np.array_equal(result.view(np.uint32), expected_bits)


@pytest.mark.parametrize("mathop", [MathOperation.Fmod, MathOperation.Remainder])
def test_scalar_modulo_fixed_schedule_ftz_observation(mathop):
    divisor = float(np.finfo(np.float32).tiny)
    input_bit_values = [
        0x00800000,
        0x00800001,
        0x00800002,
        0x00800100,
        0x00FFFFFF,
        0x01000000,
        0x01000001,
    ]
    result, src = _run(_values_from_bits(input_bit_values), mathop, divisor)
    result_bits = result.view(np.uint32)
    source_bits = src.view(np.uint32)
    expected_bits = _exact_expected_bits(src, divisor, mathop)

    observations = {}
    for input_bits in input_bit_values:
        lane_mask = source_bits == input_bits
        observed = sorted(set(int(bits) for bits in result_bits[lane_mask]))
        exact = sorted(set(int(bits) for bits in expected_bits[lane_mask]))
        observations[f"0x{input_bits:08x}"] = {
            "exact": [f"0x{bits:08x}" for bits in exact],
            "observed": [f"0x{bits:08x}" for bits in observed],
        }
        assert len(observed) == 1
        if all(bits == 0 or bits >= 0x00800000 for bits in exact):
            assert observed == exact
    print(f"Blackhole/Wormhole FTZ observation: {observations}")


@pytest.mark.parametrize("input_bits", [0x00800001, 0x00FFFFFF])
def test_scalar_modulo_fixed_schedule_ftz_raw_diagnostic(input_bits):
    divisor = float(np.finfo(np.float32).tiny)
    input_value = _values_from_bits([input_bits])[0]
    result, _ = _run([input_value] * (16 * 16), MathOperation.Fmod, divisor, diagnostic=True)
    counts = collections.Counter(int(bits) for bits in result.view(np.uint32))
    normalized_counts = {f"0x{bits:08x}": count // 32 for bits, count in sorted(counts.items()) if count >= 32}
    print(f"Blackhole/Wormhole raw FTZ diagnostic input=0x{input_bits:08x}: {normalized_counts}")
    assert sum(normalized_counts.values()) == 32
