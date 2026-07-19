# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import collections
import math
import struct
from dataclasses import dataclass

import numpy as np
import pytest
import torch
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
    3.0: (50331656.0, -4.0, -1.0, 2.0),
    5.0: (83886104.0, -6.0, -1.0, 4.0),
    7.0: (116040560.0, -8.0, -1.0, 6.0),
    10.0: (167772208.0, -12.0, -2.0, 8.0),
}
FUNCTIONAL_COUNTEREXAMPLES = {
    **{divisor: values[0] for divisor, values in COUNTEREXAMPLES.items()},
    float(np.float32(0.1)): 1677722.125,
    float(np.float32(0.3)): 5033166.5,
}


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _from_bits(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def _constants(scalar: float) -> tuple[int, int, int]:
    divisor = np.float32(abs(scalar))
    divisor_bits = _bits(float(divisor))
    reciprocal = np.float32(np.float32(1.0) / divisor)
    divisor_high_bits = divisor_bits & ~0xFFF
    return _bits(scalar), _bits(float(reciprocal)), divisor_high_bits


@dataclass
class ScalarModuloConstants(TemplateParameter):
    scalar: float

    def convert_to_cpp(self) -> str:
        divisor, reciprocal, divisor_high = _constants(self.scalar)
        return "\n".join(
            (
                f"constexpr std::uint32_t SCALAR_DIVISOR = {divisor}u;",
                f"constexpr std::uint32_t SCALAR_RECIPROCAL = {reciprocal}u;",
                f"constexpr std::uint32_t SCALAR_DIVISOR_HIGH = {divisor_high}u;",
            )
        )


def _run(
    values: list[float],
    mathop: MathOperation,
    diagnostic: bool,
    scalar: float,
):
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
        "sources/sfpu_scalar_modulo_chunked_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            FAST_MODE(FastMode.Yes if diagnostic else FastMode.No),
            MATH_OP(mathop=mathop),
            ScalarModuloConstants(scalar),
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


@pytest.mark.parametrize("divisor", [3.0, 5.0, 7.0, 10.0])
def test_scalar_modulo_chunked_raw_bits(divisor):
    dividend, residual_before, invalid_after, exact_remainder = COUNTEREXAMPLES[divisor]
    result, _ = _run(
        [dividend] * (16 * 16),
        MathOperation.Fmod,
        diagnostic=True,
        scalar=divisor,
    )

    divisor_bits, reciprocal_bits, divisor_high_bits = _constants(divisor)
    reciprocal_up_bits = reciprocal_bits + 2
    reciprocal = _from_bits(reciprocal_bits)
    scaled = np.float32(np.float32(dividend) * reciprocal)
    quotient = np.float32(math.floor(float(scaled)))
    divisor_high = _from_bits(divisor_high_bits)
    divisor_low = np.float32(np.float32(divisor) - divisor_high)

    outputs = [
        np.float32(dividend),
        np.float32(divisor),
        reciprocal,
        _from_bits(reciprocal_up_bits),
        scaled,
        quotient,
        np.float32(residual_before),
        np.float32(residual_before),
        np.float32(invalid_after),
        np.float32(exact_remainder),
        np.float32(0.0),
        divisor_high,
        divisor_low,
        *([np.float32(0.0)] * 19),
    ]
    expected_counts = collections.Counter(_bits(float(value)) for value in outputs)
    expected_counts = {bits: count * 32 for bits, count in expected_counts.items()}
    actual_counts = collections.Counter(result.view(np.uint32).tolist())
    assert actual_counts == expected_counts


@pytest.mark.parametrize("mathop", [MathOperation.Fmod, MathOperation.Remainder])
@pytest.mark.parametrize("scalar_sign", [1.0, -1.0])
@pytest.mark.parametrize("divisor", list(FUNCTIONAL_COUNTEREXAMPLES))
def test_scalar_modulo_chunked_adversarial(mathop, scalar_sign, divisor):
    counterexample = FUNCTIONAL_COUNTEREXAMPLES[divisor]
    center = np.float32(np.float32(2.0**24) * np.float32(divisor))
    values = [
        0.0,
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

    scalar = scalar_sign * divisor
    result, src = _run(values, mathop, diagnostic=False, scalar=scalar)
    source = torch.from_numpy(src.copy())
    scalar_tensor = torch.tensor(scalar, dtype=torch.float32)
    expected = (
        torch.fmod(source, scalar_tensor) if mathop == MathOperation.Fmod else torch.remainder(source, scalar_tensor)
    )
    assert np.array_equal(result.view(np.uint32), expected.numpy().view(np.uint32))
