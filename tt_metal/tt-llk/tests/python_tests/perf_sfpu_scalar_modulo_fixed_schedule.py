# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MATH_ISOLATE gate for the test-only scalar-specialized fixed schedule."""

from dataclasses import dataclass

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
from test_sfpu_scalar_modulo_fixed_schedule import FixedScheduleConstants


@dataclass
class ActiveLanes(TemplateParameter):
    active_lanes: int

    def convert_to_cpp(self) -> str:
        return f"constexpr int ACTIVE_LANES = {self.active_lanes};"


@pytest.mark.perf
@pytest.mark.parametrize("active_lanes", [0, 1, 16, 32])
@pytest.mark.parametrize("mathop", [MathOperation.Fmod, MathOperation.Remainder])
def test_perf_sfpu_scalar_modulo_fixed_schedule(perf_report, mathop, active_lanes):
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    tile_count = 8

    configuration = PerfConfig(
        "sources/sfpu_scalar_modulo_fixed_schedule_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            MATH_OP(mathop=mathop),
            FixedScheduleConstants(3.0),
            ActiveLanes(active_lanes),
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
