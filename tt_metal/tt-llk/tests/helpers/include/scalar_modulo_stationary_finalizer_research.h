// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu
{

constexpr int scalar_modulo_stationary_working_exponent = 111;

enum class ScalarModuloKind : std::uint32_t
{
    Fmod,
    FloorRemainder,
};

inline void init_scalar_modulo_stationary_finalizer_research(const std::uint32_t normalized_divisor)
{
    sfpi::vConstFloatPrgm0 = Converter::as_float(normalized_divisor);
}

// Integer round-to-nearest-even pack of a nonnegative normal value after an
// exact power-of-two scale.  Zero is accepted as a separate exact case.  The
// return value is an unsigned raw FP32 encoding; no floating store is used.
template <int DIVISOR_EXPONENT>
inline sfpi::vUInt pack_stationary_normalized_rne(const sfpi::vFloat& normalized_value)
{
    static_assert(DIVISOR_EXPONENT >= -126 && DIVISOR_EXPONENT <= 127);
    constexpr int physical_exponent_shift = DIVISOR_EXPONENT - scalar_modulo_stationary_working_exponent;
    constexpr int min_normalized_exponent = -126 - physical_exponent_shift;

    sfpi::vUInt output_bits = 0u;
    v_if (normalized_value != 0.0f)
    {
        const sfpi::vInt normalized_exponent = sfpi::exexp(normalized_value);
        const auto pack_normal               = [&normalized_value, &normalized_exponent]()
        {
            // Relative to min_normalized_exponent, biased exponent one is the
            // smallest physical normal.  Construct the raw exponent field
            // separately so no FP exponent transport or integer carry can
            // perturb the mantissa.
            const sfpi::vUInt result_biased_exponent = sfpi::as<sfpi::vUInt>(normalized_exponent - min_normalized_exponent + 1);
            return (result_biased_exponent << 23u) | (sfpi::as<sfpi::vUInt>(normalized_value) & 0x007FFFFFu);
        };

        if constexpr (min_normalized_exponent <= -126)
        {
            output_bits = pack_normal();
        }
        else
        {
            sfpi::vUInt normalized_bits = sfpi::as<sfpi::vUInt>(normalized_value);
            v_if (normalized_exponent < min_normalized_exponent)
            {
                const sfpi::vUInt shift(min_normalized_exponent - normalized_exponent);

                // A normal FP32 significand has 24 bits.  shift==24 contains
                // the tie-to-zero boundary; every shift above 24 is strictly
                // below halfway and remains the zero initialized above.
                v_if (shift <= 24u)
                {
                    normalized_bits             = (normalized_bits & 0x007FFFFFu) | 0x00800000u;
                    sfpi::vUInt truncated       = normalized_bits >> shift;
                    const sfpi::vUInt discarded = normalized_bits - (truncated << shift);
                    const sfpi::vUInt half      = sfpi::vUInt(0x00800000u) >> (sfpi::vUInt(24u) - shift);

                    v_if (discarded > half)
                    {
                        truncated += sfpi::vUInt(1u);
                    }
                    v_elseif (discarded == half && (truncated & 1u) != 0u)
                    {
                        truncated += sfpi::vUInt(1u);
                    }
                    v_endif;
                    output_bits = truncated;
                }
                v_endif;
            }
            v_else
            {
                output_bits = pack_normal();
            }
            v_endif;
        }
    }
    v_endif;
    return output_bits;
}

template <ScalarModuloKind KIND, int DIVISOR_EXPONENT, bool DIVIDEND_NEGATIVE, bool DIVISOR_NEGATIVE>
inline sfpi::vUInt finalize_stationary_result(const sfpi::vFloat& normalized_r)
{
    constexpr bool signs_differ = DIVIDEND_NEGATIVE != DIVISOR_NEGATIVE;

    sfpi::vFloat magnitude = normalized_r;
    if constexpr (KIND == ScalarModuloKind::FloorRemainder && signs_differ)
    {
        // The exact-zero predicate is intentionally evaluated before D-R.
        v_if (normalized_r != 0.0f)
        {
            magnitude = sfpi::abs(sfpi::vConstFloatPrgm0) - normalized_r;
        }
        v_endif;
    }

    sfpi::vUInt output_bits = pack_stationary_normalized_rne<DIVISOR_EXPONENT>(magnitude);

    if constexpr (KIND == ScalarModuloKind::Fmod)
    {
        if constexpr (DIVIDEND_NEGATIVE)
        {
            output_bits |= 0x80000000u;
        }
    }
    else
    {
        if constexpr (DIVISOR_NEGATIVE)
        {
            output_bits |= 0x80000000u;
        }
        if constexpr (signs_differ)
        {
            // Nonzero floor remainders use the divisor sign.  Exact zero uses
            // the dividend sign, so differing signs require one zero-only flip.
            v_if (normalized_r == 0.0f)
            {
                output_bits ^= 0x80000000u;
            }
            v_endif;
        }
    }
    return output_bits;
}

template <ScalarModuloKind KIND, int DIVISOR_EXPONENT, bool DIVIDEND_NEGATIVE, bool DIVISOR_NEGATIVE, int ITERATIONS = 32>
inline void calculate_scalar_modulo_stationary_finalizer()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat normalized_r = sfpi::dst_reg[0];
        sfpi::dst_reg[0]                = finalize_stationary_result<KIND, DIVISOR_EXPONENT, DIVIDEND_NEGATIVE, DIVISOR_NEGATIVE>(normalized_r);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
