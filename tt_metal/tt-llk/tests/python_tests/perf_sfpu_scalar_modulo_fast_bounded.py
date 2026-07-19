# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Identical MATH_ISOLATE comparison of FastBoundedExactBH and Robust."""

from dataclasses import dataclass
from enum import IntEnum

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathOperation, PerfRunType
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TILE_COUNT,
    TemplateParameter,
)
from test_sfpu_scalar_modulo_fast_bounded import WORKING_EXPONENT, _working_constants
from test_sfpu_scalar_modulo_stationary_combined import _constants, _normal_parts


class FastBoundedPerfPath(IntEnum):
    FAST = 0
    ROBUST = 1


PERF_DIVISORS = [
    (0x40400000, "three"),
    (0x00800000, "smallest_normal"),
    (0x00800001, "smallest_normal_next"),
    (0x0C000000, "eb_minus_103"),
    (0x0B800000, "eb_minus_104"),
    (0x77000000, "eb_111"),
    (0x77800000, "eb_112"),
    (0x7F7FFFFF, "flt_max"),
]


@dataclass
class FastBoundedPerfConstants(TemplateParameter):
    path: FastBoundedPerfPath
    divisor_bits: int

    def convert_to_cpp(self) -> str:
        fast_normalized, _, fast_reciprocal, fast_high, reciprocal_bias = (
            _working_constants(self.divisor_bits)
        )
        robust_normalized, robust_reciprocal, start, initial, robust_high = _constants(
            self.divisor_bits
        )
        _, _, divisor_exponent = _normal_parts(self.divisor_bits)
        assert fast_normalized == robust_normalized
        assert fast_high == robust_high
        return "\n".join(
            (
                f"constexpr int FAST_BOUNDED_PERF_PATH = {int(self.path)};",
                f"constexpr std::uint32_t SCALAR_PHYSICAL_DIVISOR = {self.divisor_bits}u;",
                f"constexpr std::uint32_t SCALAR_NORMALIZED_DIVISOR = {fast_normalized}u;",
                f"constexpr std::uint32_t FAST_WORKING_RECIPROCAL = {fast_reciprocal}u;",
                f"constexpr std::uint32_t SCALAR_ROBUST_RECIPROCAL_UP = {robust_reciprocal}u;",
                f"constexpr int SCALAR_DIVISOR_EXPONENT = {divisor_exponent};",
                f"constexpr int FAST_INPUT_EXPONENT_SHIFT = {WORKING_EXPONENT - divisor_exponent};",
                f"constexpr unsigned FAST_HIGH_MANTISSA = {fast_high}u;",
                f"constexpr unsigned FAST_RECIPROCAL_ULP_BIAS = {reciprocal_bias}u;",
                f"constexpr int SCALAR_START_SHIFT = {start};",
                f"constexpr int SCALAR_INITIAL_SHIFT = {initial};",
                f"constexpr unsigned SCALAR_ROBUST_HIGH_MANTISSA = {robust_high}u;",
            )
        )


@pytest.mark.perf
@pytest.mark.parametrize(
    ("divisor_bits", "divisor_name"),
    PERF_DIVISORS,
    ids=[name for _, name in PERF_DIVISORS],
)
@pytest.mark.parametrize(
    "path", list(FastBoundedPerfPath), ids=lambda path: path.name.lower()
)
@pytest.mark.parametrize("mathop", [MathOperation.Fmod, MathOperation.Remainder])
def test_perf_sfpu_scalar_modulo_fast_bounded(
    perf_report, mathop, path, divisor_bits, divisor_name
):
    del divisor_name
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    tile_count = 8

    configuration = PerfConfig(
        "sources/sfpu_scalar_modulo_fast_bounded_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            FastBoundedPerfConstants(path, divisor_bits),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(16),
            NUM_FACES(num_faces=4),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.Yes,
        compile_time_formats=True,
    )
    configuration.run(perf_report)
