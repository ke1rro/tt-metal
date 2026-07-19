// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "scalar_modulo_stationary_finalizer_research.h"
#include "scalar_modulo_stationary_research.h"

namespace ckernel::sfpu
{

inline void init_scalar_modulo_stationary_combined_research(
    const std::uint32_t normalized_divisor, const std::uint32_t reciprocal_up, const std::uint32_t physical_divisor)
{
    sfpi::vConstFloatPrgm0 = Converter::as_float(normalized_divisor);
    sfpi::vConstFloatPrgm1 = Converter::as_float(reciprocal_up);
    sfpi::vConstFloatPrgm2 = Converter::as_float(physical_divisor);
}

template <int START_SHIFT, int INITIAL_SHIFT, unsigned DIVISOR_HIGH_MANTISSA>
inline sfpi::vFloat reduce_scalar_modulo_stationary_combined(const sfpi::vFloat& input_magnitude)
{
    static_assert(INITIAL_SHIFT >= -16 && INITIAL_SHIFT <= 0);

    sfpi::vFloat normalized_r = input_magnitude;
    v_if (sfpi::exexp(input_magnitude) == 127)
    {
        normalized_r = sfpi::addexp(normalized_r, -1);
    }
    v_endif;

    if constexpr (INITIAL_SHIFT != 0)
    {
        v_if (normalized_r != 0.0f)
        {
            normalized_r = sfpi::addexp(normalized_r, INITIAL_SHIFT);
        }
        v_endif;
    }

    scalar_modulo_stationary_reduce_inplace<START_SHIFT, DIVISOR_HIGH_MANTISSA>(normalized_r);

    // Exact reconstruction for the mandatory exponent-127 pre-half:
    // a mod D = (2 * ((a/2) mod D)) mod D.
    v_if (sfpi::exexp(input_magnitude) == 127 && normalized_r != 0.0f)
    {
        normalized_r = sfpi::addexp(normalized_r, 1);
        v_if (normalized_r >= sfpi::abs(sfpi::vConstFloatPrgm0))
        {
            normalized_r = normalized_r - sfpi::abs(sfpi::vConstFloatPrgm0);
        }
        v_endif;
    }
    v_endif;

    return normalized_r;
}

template <ScalarModuloKind KIND, bool SIGNS_DIFFER>
inline sfpi::vUInt finalize_high_divisor_bypass_magnitude(const sfpi::vFloat& input_magnitude)
{
    sfpi::vUInt output_bits = sfpi::as<sfpi::vUInt>(input_magnitude);

    if constexpr (KIND == ScalarModuloKind::FloorRemainder && SIGNS_DIFFER)
    {
        v_if (input_magnitude != 0.0f)
        {
            output_bits = sfpi::as<sfpi::vUInt>(sfpi::abs(sfpi::vConstFloatPrgm2) - input_magnitude);
        }
        v_endif;
    }
    return output_bits;
}

template <ScalarModuloKind KIND, int DIVISOR_EXPONENT, bool SIGNS_DIFFER>
inline sfpi::vUInt finalize_stationary_magnitude(const sfpi::vFloat& normalized_r)
{
    sfpi::vFloat magnitude = normalized_r;
    if constexpr (KIND == ScalarModuloKind::FloorRemainder && SIGNS_DIFFER)
    {
        v_if (normalized_r != 0.0f)
        {
            magnitude = sfpi::abs(sfpi::vConstFloatPrgm0) - normalized_r;
        }
        v_endif;
    }
    return pack_stationary_normalized_rne<DIVISOR_EXPONENT>(magnitude);
}

template <ScalarModuloKind KIND, bool DIVIDEND_NEGATIVE, bool DIVISOR_NEGATIVE>
inline sfpi::vUInt restore_stationary_sign(sfpi::vUInt positive_bits)
{
    constexpr bool signs_differ = DIVIDEND_NEGATIVE != DIVISOR_NEGATIVE;

    if constexpr (KIND == ScalarModuloKind::Fmod)
    {
        if constexpr (DIVIDEND_NEGATIVE)
        {
            positive_bits |= 0x80000000u;
        }
    }
    else if constexpr (signs_differ)
    {
        v_if (positive_bits == 0u)
        {
            if constexpr (DIVIDEND_NEGATIVE)
            {
                positive_bits |= 0x80000000u;
            }
        }
        v_else
        {
            if constexpr (DIVISOR_NEGATIVE)
            {
                positive_bits |= 0x80000000u;
            }
        }
        v_endif;
    }
    else if constexpr (DIVISOR_NEGATIVE)
    {
        positive_bits |= 0x80000000u;
    }
    return positive_bits;
}

template <
    ScalarModuloKind KIND,
    int DIVISOR_EXPONENT,
    bool DIVIDEND_NEGATIVE,
    bool DIVISOR_NEGATIVE,
    int START_SHIFT,
    int INITIAL_SHIFT,
    unsigned DIVISOR_HIGH_MANTISSA,
    int ITERATIONS = 32>
inline void calculate_scalar_modulo_stationary_combined()
{
    static_assert(DIVISOR_EXPONENT >= -126 && DIVISOR_EXPONENT <= 127);

#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat input_magnitude = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[0]));
        if constexpr (DIVISOR_EXPONENT > scalar_modulo_stationary_working_exponent)
        {
            // Scaling a tiny a<b input down into the exponent-111 frame can
            // underflow before reduction.  Keep bypass and reduction in
            // disjoint predicated paths so no bypass register remains live
            // through the reducer/finalizer phase.
            sfpi::vUInt output_bits = 0u;
            v_if (input_magnitude < sfpi::abs(sfpi::vConstFloatPrgm2))
            {
                output_bits = finalize_high_divisor_bypass_magnitude<KIND, (DIVIDEND_NEGATIVE != DIVISOR_NEGATIVE)>(input_magnitude);
            }
            v_else
            {
                sfpi::vFloat normalized_r;
                {
                    normalized_r = reduce_scalar_modulo_stationary_combined<START_SHIFT, INITIAL_SHIFT, DIVISOR_HIGH_MANTISSA>(input_magnitude);
                }
                {
                    output_bits = finalize_stationary_magnitude<KIND, DIVISOR_EXPONENT, (DIVIDEND_NEGATIVE != DIVISOR_NEGATIVE)>(normalized_r);
                }
            }
            v_endif;
            sfpi::dst_reg[0] = restore_stationary_sign<KIND, DIVIDEND_NEGATIVE, DIVISOR_NEGATIVE>(output_bits);
        }
        else
        {
            sfpi::vFloat normalized_r;
            {
                normalized_r = reduce_scalar_modulo_stationary_combined<START_SHIFT, INITIAL_SHIFT, DIVISOR_HIGH_MANTISSA>(input_magnitude);
            }
            {
                sfpi::dst_reg[0] = finalize_stationary_result<KIND, DIVISOR_EXPONENT, DIVIDEND_NEGATIVE, DIVISOR_NEGATIVE>(normalized_r);
            }
        }
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
