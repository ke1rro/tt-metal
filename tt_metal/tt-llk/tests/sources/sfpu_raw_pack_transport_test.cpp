// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test-only probe for the SFPU -> DEST -> packer -> L1 raw-word path.
//
// The SFPU synthesizes the probe words directly in an LReg with integer
// SFPLOADI instructions, avoiding any input unpack/datacopy conversion.  The
// output buffer uses UInt32 so the host never numerically converts the words.
// The packer is explicitly configured as UInt32 -> UInt32.  STORE_AS_FP32
// selects the FP32 control path.  The raw path uses BH's working INT32 store;
// on WH it exercises the strongest documented opaque-32 alternative (LO16
// with a compensating half rotation).  The host test records whether the
// complete architecture path preserves or flushes FP32 subnormal encodings.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu
{

// llk_defs::InstrModLoadStore names mode 6 as LO16 for load-side
// compatibility, but SFPSTORE's opaque half-rotating LO16 mode is numeric 9
// and currently has no correctly named enum member.
static constexpr unsigned SFPSTORE_MOD0_FMT_LO16_OPAQUE = 9;

template <bool STORE_AS_FP32_, std::uint32_t WORD>
inline void store_raw_pack_probe_word()
{
#if __riscv_xtttensixwh
    constexpr std::uint32_t stored_word = STORE_AS_FP32_ ? WORD : ((WORD << 16) | (WORD >> 16));
#else
    constexpr std::uint32_t stored_word = WORD;
#endif
    const sfpi::vUInt bits(stored_word);
    if constexpr (STORE_AS_FP32_)
    {
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(bits);
    }
    else
    {
#if __riscv_xtttensixwh
        // WH's INT32 store flushes FP32 subnormal encodings.  LO16 is an
        // opaque-32 store with a 16-bit half rotation; pre-rotating above
        // makes the two rotations cancel without any floating conversion.
        __builtin_rvtt_sfpstore(bits.get(), sfpi::dst_reg[0].get(), SFPSTORE_MOD0_FMT_LO16_OPAQUE, sfpi::SFPSTORE_ADDR_MODE_NOINC);
#else
        sfpi::dst_reg[0] = bits;
#endif
    }
    sfpi::dst_reg++;
}

template <bool STORE_AS_FP32_>
inline void calculate_raw_pack_transport()
{
    // Re-establish an unconditional, writable lane state at the point of use.
    // This is intentionally after the 32-bit unpack-to-dest handshake: on WH,
    // initializing SFPU only in the kernel prelude left every subsequent store
    // ineffective on silicon.  The NOP is the ISA-required LaneConfig settle
    // slot when DISABLE_BACKDOOR_LOAD may have changed.
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPNOP;
    TTI_SFPENCC(0, 0, 0, 2);

    // One broadcast word per SFPU row.  The first 24 rows cover signed
    // zeros, subnormal threshold encodings, min-normal neighbors and ordinary
    // normals; the remaining rows are deterministic zero padding.  SFPI's
    // vUInt constructor lowers each exact 32-bit constant to SFPLOADI.
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x80000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000001u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000002u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000003u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x003FFFFEu>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x003FFFFFu>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00400000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00400001u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x007FFFFEu>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x007FFFFFu>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x80000001u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x803FFFFFu>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x80400000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x807FFFFFu>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00800000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00800001u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x80800000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x80800001u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x3F000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x3F800000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0xBF800000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x7F7FFFFFu>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0xFF7FFFFFu>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
    store_raw_pack_probe_word<STORE_AS_FP32_, 0x00000000u>();
}

} // namespace ckernel::sfpu

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr bool is_int_fpu_en = true;

    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(TILE_NUM_FACES, formats.math);
    SFPU_UNARY_INIT(unused);

    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");
    LLK_ASSERT((params.NUM_BLOCKS == 1 && params.NUM_TILES_IN_BLOCK == 1), "raw pack transport probe supports exactly one tile");

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, formats.math, formats.math);
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_raw_pack_transport, (STORE_AS_FP32), tile, VectorMode::None);
        }
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Keep the complete path unsigned: unlike the historical Int32 hash
    // transport, UInt32 must treat every payload as an opaque four-byte word.
    constexpr std::uint32_t raw_format = static_cast<std::uint32_t>(DataFormat::UInt32);
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        raw_format, raw_format, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES * sizeof(std::uint32_t), FACE_R_DIM, TILE_NUM_FACES);

    _llk_pack_init_wrapper_<PackMode::Default, false>(raw_format, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[block * params.NUM_TILES_IN_BLOCK + tile]));
        }
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
