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
#include "sfpu_scalar_modulo_hybrid.h"

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
        ckernel::sfpu::init_scalar_modulo_research(0x40400000u, 0x3eaaaaabu);
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
                    if constexpr (SFPU_UNARY_OPERATION == SfpuType::fmod)
                    {
                        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_scalar_modulo_hybrid, (false, 32), block_tile, ckernel::VectorMode::None);
                    }
                    else
                    {
                        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_scalar_modulo_hybrid, (true, 32), block_tile, ckernel::VectorMode::None);
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
