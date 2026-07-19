// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu
{

constexpr int scalar_modulo_stationary_chunk_bits = 16;
constexpr int scalar_modulo_stationary_step       = scalar_modulo_stationary_chunk_bits - 1;

inline void init_scalar_modulo_stationary_research(const std::uint32_t normalized_divisor, const std::uint32_t reciprocal_up)
{
    sfpi::vConstFloatPrgm0 = Converter::as_float(normalized_divisor);
    sfpi::vConstFloatPrgm1 = Converter::as_float(reciprocal_up);
}

// Test-only normalized-magnitude contract:
//   * input and physical scalar divisor are finite normal FP32 magnitudes, or
//     the input is zero;
//   * normalized_divisor is the scalar significand at unbiased exponent 111;
//   * reciprocal_up is nextUp(nextUp(RN32(1/normalized_divisor))), except an
//     exact power-of-two reciprocal is left exact;
//   * every active local quotient is below 2**16 and its converted estimate is
//     either the exact quotient or that quotient plus one;
//   * DIVISOR_HIGH_MANTISSA selects the top 12 significand bits, so each
//     UINT16-by-12-bit partial product fits SFPMAD's 28-bit product precision;
//   * the result remains in the exponent-111 frame.  Sign restoration and the
//     physical normal/subnormal pack are deliberately outside this prototype.
template <unsigned DIVISOR_HIGH_MANTISSA>
inline void scalar_modulo_stationary_stage(sfpi::vFloat& residual, const sfpi::vFloat& normalized_divisor, const sfpi::vFloat& reciprocal_up)
{
    v_if (residual >= normalized_divisor)
    {
        const sfpi::vFloat scaled_quotient   = residual * reciprocal_up;
        const sfpi::vUInt16 quotient_integer = sfpi::convert<sfpi::vUInt16>(scaled_quotient, sfpi::RoundMode::Nearest);
        const sfpi::vFloat quotient          = sfpi::convert<sfpi::vFloat>(quotient_integer, sfpi::RoundMode::Nearest);

        const sfpi::vFloat divisor_high = sfpi::setman(normalized_divisor, DIVISOR_HIGH_MANTISSA);
        const sfpi::vFloat divisor_low  = normalized_divisor - divisor_high;
        residual                        = residual - quotient * divisor_high;
        residual                        = residual - quotient * divisor_low;

        v_if (residual < 0.0f)
        {
            residual = residual + normalized_divisor;
        }
        v_endif;
    }
    v_endif;
}

template <int START_SHIFT, unsigned DIVISOR_HIGH_MANTISSA>
inline void scalar_modulo_stationary_reduce_inplace(sfpi::vFloat& residual)
{
    static_assert(START_SHIFT >= 0 && START_SHIFT <= 237);
    static_assert(DIVISOR_HIGH_MANTISSA <= 0xFFFu);
    static_assert((DIVISOR_HIGH_MANTISSA & 1u) == 0u);

    constexpr int full_decrements = START_SHIFT / scalar_modulo_stationary_step;
    constexpr int final_decrement = START_SHIFT % scalar_modulo_stationary_step;

    const sfpi::vFloat normalized_divisor = sfpi::abs(sfpi::vConstFloatPrgm0);
    const sfpi::vFloat reciprocal_up      = sfpi::vConstFloatPrgm1;

#pragma GCC unroll 0
    for (int stage = 0; stage < full_decrements; ++stage)
    {
        scalar_modulo_stationary_stage<DIVISOR_HIGH_MANTISSA>(residual, normalized_divisor, reciprocal_up);
        v_if (residual != 0.0f)
        {
            residual = sfpi::addexp(residual, scalar_modulo_stationary_step);
        }
        v_endif;
    }

    if constexpr (final_decrement != 0)
    {
        scalar_modulo_stationary_stage<DIVISOR_HIGH_MANTISSA>(residual, normalized_divisor, reciprocal_up);
        v_if (residual != 0.0f)
        {
            residual = sfpi::addexp(residual, final_decrement);
        }
        v_endif;
    }

    scalar_modulo_stationary_stage<DIVISOR_HIGH_MANTISSA>(residual, normalized_divisor, reciprocal_up);
}

template <int START_SHIFT, int INITIAL_SHIFT, unsigned DIVISOR_HIGH_MANTISSA, int ITERATIONS = 32>
inline void calculate_scalar_modulo_stationary_normalized()
{
    static_assert(INITIAL_SHIFT >= -16 && INITIAL_SHIFT <= 0);

#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        sfpi::vFloat residual;
        {
            const sfpi::vFloat input           = sfpi::dst_reg[0];
            const sfpi::vFloat input_magnitude = sfpi::abs(input);
            residual                           = input_magnitude;
            v_if (sfpi::exexp(input_magnitude) == 127)
            {
                residual = sfpi::addexp(residual, -1);
            }
            v_endif;
        }

        if constexpr (INITIAL_SHIFT != 0)
        {
            v_if (residual != 0.0f)
            {
                residual = sfpi::addexp(residual, INITIAL_SHIFT);
            }
            v_endif;
        }

        scalar_modulo_stationary_reduce_inplace<START_SHIFT, DIVISOR_HIGH_MANTISSA>(residual);

        // Exact reconstruction for the mandatory exponent-127 pre-half:
        // a mod D = (2 * ((a/2) mod D)) mod D.
        {
            const sfpi::vFloat input           = sfpi::dst_reg[0];
            const sfpi::vFloat input_magnitude = sfpi::abs(input);
            v_if (sfpi::exexp(input_magnitude) == 127 && residual != 0.0f)
            {
                residual = sfpi::addexp(residual, 1);
                v_if (residual >= sfpi::abs(sfpi::vConstFloatPrgm0))
                {
                    residual = residual - sfpi::abs(sfpi::vConstFloatPrgm0);
                }
                v_endif;
            }
            v_endif;
        }

        sfpi::dst_reg[0] = residual;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
