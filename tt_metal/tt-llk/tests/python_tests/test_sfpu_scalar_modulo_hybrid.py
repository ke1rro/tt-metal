# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

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
    generate_input_dim,
)


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _run(values: list[float], mathop: MathOperation, diagnostic: bool):
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
        "sources/sfpu_scalar_modulo_hybrid_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            FAST_MODE(FastMode.Yes if diagnostic else FastMode.No),
            MATH_OP(mathop=mathop),
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


def test_scalar_modulo_blackhole_diagnostic():
    result, _ = _run([50331656.0], MathOperation.Fmod, diagnostic=True)

    # The full-tile SFPU loop emits one checkpoint per vector iteration. Each
    # checkpoint is constant across its 32 lanes, although face packing may
    # reorder those vectors. Validate the exact normal FP32 values as raw bits.
    unique_bits, counts = np.unique(result.view(np.uint32), return_counts=True)
    bit_counts = dict(zip(unique_bits.tolist(), counts.tolist()))
    required = {
        _bits(50331656.0): 32,  # a and initial scaled divisor share this value
        _bits(3.0): 32,
        0x3EAAAAAB: 32,
        0x4B800002: 64,  # scaled and q_hat
        _bits(-4.0): 32,
        _bits(0.0): 32,  # classifier: unsafe
        _bits(-1.0): 32,  # invalid two-correction result
        _bits(2.0): 64,  # final magnitude and signed result
    }
    for bits, minimum_count in required.items():
        assert bit_counts.get(bits, 0) >= minimum_count, (
            f"missing diagnostic 0x{bits:08x}: " f"found {bit_counts.get(bits, 0)}, expected at least {minimum_count}"
        )


@pytest.mark.parametrize("mathop", [MathOperation.Fmod, MathOperation.Remainder])
def test_scalar_modulo_hybrid_adversarial(mathop):
    values = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        50331656.0,
        -50331656.0,
        50855936.0,
        -50855936.0,
        2.0**22,
        -(2.0**22),
        float(np.nextafter(np.float32(3.0 * (2.0**24)), np.float32(-np.inf))),
        float(np.nextafter(np.float32(3.0 * (2.0**24)), np.float32(np.inf))),
    ]
    result, src = _run(values, mathop, diagnostic=False)
    source = torch.from_numpy(src.copy())
    divisor = torch.tensor(3.0, dtype=torch.float32)
    expected = torch.fmod(source, divisor) if mathop == MathOperation.Fmod else torch.remainder(source, divisor)

    assert np.array_equal(result.view(np.uint32), expected.numpy().view(np.uint32))
