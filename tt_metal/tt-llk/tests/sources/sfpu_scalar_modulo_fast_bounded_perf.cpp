// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST   = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        _perf_unpack_loop_set_valid<true, is_fp32_dest_acc_en>(num_faces * TILE_CNT * LOOP_FACTOR);
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "scalar_modulo_fast_bounded_research.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    constexpr auto data_copy_type = ckernel::DataCopyType::A2D;

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        SFPU_UNARY_INIT(unused);
        if constexpr (FAST_BOUNDED_PERF_PATH == 0)
        {
            ckernel::sfpu::init_scalar_modulo_fast_bounded_research(SCALAR_NORMALIZED_DIVISOR, FAST_WORKING_RECIPROCAL, SCALAR_PHYSICAL_DIVISOR);
        }
        else
        {
            ckernel::sfpu::init_scalar_modulo_stationary_combined_research(SCALAR_NORMALIZED_DIVISOR, SCALAR_ROBUST_RECIPROCAL_UP, SCALAR_PHYSICAL_DIVISOR);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
            {
                const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                {
                    LLK_ASSERT(
                        (block_tile < ckernel::get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>()),
                        "block_tile exceeds max dest tiles");
                    _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::BroadcastType::NONE, unpack_to_dest>(
                        block_tile, formats.math, formats.math);

                    if constexpr (FAST_BOUNDED_PERF_PATH == 0)
                    {
                        if constexpr (SFPU_UNARY_OPERATION == SfpuType::fmod)
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_scalar_modulo_fast_bounded_exact,
                                (sfpu::ScalarModuloKind::Fmod,
                                 SCALAR_DIVISOR_EXPONENT,
                                 false,
                                 false,
                                 FAST_INPUT_EXPONENT_SHIFT,
                                 FAST_HIGH_MANTISSA,
                                 FAST_RECIPROCAL_ULP_BIAS,
                                 32),
                                block_tile,
                                ckernel::VectorMode::None);
                        }
                        else
                        {
                            SFPU_UNARY_CALL(
                                DST_SYNC_MODE,
                                is_fp32_dest_acc_en,
                                calculate_scalar_modulo_fast_bounded_exact,
                                (sfpu::ScalarModuloKind::FloorRemainder,
                                 SCALAR_DIVISOR_EXPONENT,
                                 false,
                                 true,
                                 FAST_INPUT_EXPONENT_SHIFT,
                                 FAST_HIGH_MANTISSA,
                                 FAST_RECIPROCAL_ULP_BIAS,
                                 32),
                                block_tile,
                                ckernel::VectorMode::None);
                        }
                    }
                    else if constexpr (SFPU_UNARY_OPERATION == SfpuType::fmod)
                    {
                        SFPU_UNARY_CALL(
                            DST_SYNC_MODE,
                            is_fp32_dest_acc_en,
                            calculate_scalar_modulo_stationary_combined,
                            (sfpu::ScalarModuloKind::Fmod,
                             SCALAR_DIVISOR_EXPONENT,
                             false,
                             false,
                             SCALAR_START_SHIFT,
                             SCALAR_INITIAL_SHIFT,
                             SCALAR_ROBUST_HIGH_MANTISSA,
                             32),
                            block_tile,
                            ckernel::VectorMode::None);
                    }
                    else
                    {
                        SFPU_UNARY_CALL(
                            DST_SYNC_MODE,
                            is_fp32_dest_acc_en,
                            calculate_scalar_modulo_stationary_combined,
                            (sfpu::ScalarModuloKind::FloorRemainder,
                             SCALAR_DIVISOR_EXPONENT,
                             false,
                             true,
                             SCALAR_START_SHIFT,
                             SCALAR_INITIAL_SHIFT,
                             SCALAR_ROBUST_HIGH_MANTISSA,
                             32),
                            block_tile,
                            ckernel::VectorMode::None);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    {
        START_PERF_MEASURE("INIT")
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        PROFILER_SYNC();
    }
}

#endif
