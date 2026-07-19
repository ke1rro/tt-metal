# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Architecture contract for SFPU-generated raw FP32 encodings."""

from dataclasses import dataclass

import numpy as np
import pytest
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
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

# Includes zero signs, tiny subnormals, subnormal threshold encodings,
# the largest subnormal, the smallest normal and ordinary normal values.
TRANSPORT_WORDS = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x00000002,
    0x00000003,
    0x003FFFFE,
    0x003FFFFF,
    0x00400000,
    0x00400001,
    0x007FFFFE,
    0x007FFFFF,
    0x80000001,
    0x803FFFFF,
    0x80400000,
    0x807FFFFF,
    0x00800000,
    0x00800001,
    0x80800000,
    0x80800001,
    0x3F000000,
    0x3F800000,
    0xBF800000,
    0x7F7FFFFF,
    0xFF7FFFFF,
]


@dataclass
class StoreMode(TemplateParameter):
    as_fp32: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool STORE_AS_FP32 = {str(self.as_fp32).lower()};"


def _skip_if_unsupported() -> None:
    if TestConfig.CHIP_ARCH not in {
        ChipArchitecture.BLACKHOLE,
        ChipArchitecture.WORMHOLE,
    }:
        pytest.skip(
            f"raw SFPU pack transport probe is unsupported on {TestConfig.CHIP_ARCH}"
        )


def _run_transport(*, store_as_fp32: bool) -> np.ndarray:
    _skip_if_unsupported()
    formats = InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32)
    input_dimensions = [32, 32]

    src_a, tile_count_a, src_b, tile_count_b = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.custom(TRANSPORT_WORDS),
    )
    configuration = TestConfig(
        "sources/sfpu_raw_pack_transport_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            StoreMode(store_as_fp32),
        ],
        runtimes=[TILE_COUNT(tile_count_a), NUM_BLOCKS(1), NUM_TILES_IN_BLOCK(1)],
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
    return result


def _synthesized_tile() -> np.ndarray:
    rows = np.asarray(
        TRANSPORT_WORDS + [0] * (32 - len(TRANSPORT_WORDS)), dtype=np.uint32
    )
    return np.repeat(rows, 32)


def _fp32_store_expected(source: np.ndarray) -> np.ndarray:
    expected = source.copy()
    subnormal = ((expected & np.uint32(0x7F800000)) == 0) & (
        (expected & np.uint32(0x007FFFFF)) != 0
    )
    expected[subnormal] &= np.uint32(0x80000000)
    return expected


def _count_diff(
    observed: np.ndarray, expected: np.ndarray
) -> dict[str, tuple[int, int]]:
    words = np.union1d(observed, expected)
    return {
        f"0x{int(word):08x}": (
            int(np.count_nonzero(observed == word)),
            int(np.count_nonzero(expected == word)),
        )
        for word in words
        if np.count_nonzero(observed == word) != np.count_nonzero(expected == word)
    }


def test_sfpu_raw_store_pack_architecture_contract():
    result = _run_transport(store_as_fp32=False)
    source = _synthesized_tile()
    if TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE:
        expected = source
    else:
        # WH silicon flushes FP32-subnormal bit patterns at the Dst/pack
        # boundary even through MOD0=9 opaque-32 store and UInt32 pack.
        expected = _fp32_store_expected(source)

    assert np.array_equal(np.sort(result), np.sort(expected)), _count_diff(
        result, expected
    )

    source_is_subnormal = ((source & np.uint32(0x7F800000)) == 0) & (
        (source & np.uint32(0x007FFFFF)) != 0
    )
    if TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE:
        # The output is intentionally transported as UInt32, but consumers can
        # reinterpret the same bytes as FP32 without losing an encoding.
        assert np.array_equal(result.view(np.float32).view(np.uint32), result)
        for word in TRANSPORT_WORDS:
            assert np.count_nonzero(result == np.uint32(word)) == np.count_nonzero(
                expected == np.uint32(word)
            )
    else:
        assert np.any(source_is_subnormal)
        assert not np.array_equal(np.sort(result), np.sort(source))
        for word in source[source_is_subnormal]:
            assert np.count_nonzero(result == word) == 0


def test_sfpu_fp32_store_control_flushes_only_subnormals():
    result = _run_transport(store_as_fp32=True)
    source = _synthesized_tile()
    expected = _fp32_store_expected(source)
    assert np.array_equal(np.sort(result), np.sort(expected)), _count_diff(
        result, expected
    )

    source_is_subnormal = ((source & np.uint32(0x7F800000)) == 0) & (
        (source & np.uint32(0x007FFFFF)) != 0
    )
    assert np.any(source_is_subnormal)
    for word in source[source_is_subnormal]:
        assert np.count_nonzero(result == word) == 0
