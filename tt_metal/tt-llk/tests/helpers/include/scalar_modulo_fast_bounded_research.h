// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "scalar_modulo_stationary_combined_research.h"

namespace ckernel::sfpu
{

constexpr int scalar_modulo_fast_bounded_working_exponent = 103;
constexpr int scalar_modulo_fast_bounded_quotient_bits    = 22;
constexpr int scalar_modulo_fast_bounded_split            = scalar_modulo_stationary_working_exponent - scalar_modulo_fast_bounded_working_exponent;

inline void init_scalar_modulo_fast_bounded_research(
    const std::uint32_t normalized_divisor, const std::uint32_t working_reciprocal, const std::uint32_t physical_divisor)
{
    sfpi::vConstFloatPrgm0 = Converter::as_float(normalized_divisor);
    sfpi::vConstFloatPrgm1 = Converter::as_float(working_reciprocal);
    sfpi::vConstFloatPrgm2 = Converter::as_float(physical_divisor);
}

template <unsigned DIVISOR_HIGH_MANTISSA, unsigned RECIPROCAL_ULP_BIAS>
inline void scalar_modulo_fast_bounded_stage(sfpi::vFloat& residual, const sfpi::vFloat& divisor, const sfpi::vFloat& reciprocal_rn)
{
    static_assert(DIVISOR_HIGH_MANTISSA <= 0xFFFu);
    static_assert((DIVISOR_HIGH_MANTISSA & 1u) == 0u);
    static_assert(RECIPROCAL_ULP_BIAS == 0u || RECIPROCAL_ULP_BIAS == 2u);

    v_if (residual >= divisor)
    {
        const sfpi::vFloat reciprocal_up     = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(reciprocal_rn) + static_cast<std::uint32_t>(RECIPROCAL_ULP_BIAS));
        const sfpi::vFloat scaled_quotient   = residual * reciprocal_up;
        const sfpi::vUInt16 quotient_integer = sfpi::convert<sfpi::vUInt16>(scaled_quotient, sfpi::RoundMode::Zero);
        const sfpi::vFloat quotient          = sfpi::convert<sfpi::vFloat>(quotient_integer, sfpi::RoundMode::Nearest);

        const sfpi::vFloat divisor_high = sfpi::setman(divisor, DIVISOR_HIGH_MANTISSA);
        const sfpi::vFloat divisor_low  = divisor - divisor_high;
        residual                        = residual - quotient * divisor_high;
        residual                        = residual - quotient * divisor_low;

        v_if (residual < 0.0f)
        {
            residual = residual + divisor;
        }
        v_endif;
    }
    v_endif;
}

// Caller contract:
//   * input and scalar divisor are finite normal FP32 magnitudes, or input is
//     zero;
//   * for input >= divisor, the actual Blackhole FP32 value
//       scaled = input_at_exponent_103 * RN32(1 / divisor_at_exponent_103)
//     is finite and has unbiased exponent below 22;
//   * caller dispatches to this kernel before launch; there is no lane-local
//     fallback to the stationary Robust kernel;
//   * the actual reducer uses two one-sided local quotient estimates.  The
//     exponent-111 first divisor bounds its exact quotient at 2**14 and its
//     estimate at 2**14 + 1.  The exponent-103 second divisor bounds its exact
//     quotient below 2**8 and its estimate at 2**8;
//   * each local subtraction splits the divisor into two at-most-12-bit
//     components, reusing the stationary stage's 28-bit SFPMAD proof shape.
//
// The exponent-103 frame keeps every bounded input/product finite and every
// divisor-lattice remainder normal.  The exact remainder is scaled by 2**8
// into the already-proven exponent-111 finalizer contract.
template <int INPUT_EXPONENT_SHIFT, unsigned DIVISOR_HIGH_MANTISSA, unsigned RECIPROCAL_ULP_BIAS>
inline sfpi::vFloat reduce_scalar_modulo_fast_bounded_exact(const sfpi::vFloat& input_magnitude)
{
    static_assert(INPUT_EXPONENT_SHIFT >= -24 && INPUT_EXPONENT_SHIFT <= 229);

    sfpi::vFloat residual = sfpi::setexp(input_magnitude, sfpi::exexp(input_magnitude, sfpi::ExponentMode::Biased) + INPUT_EXPONENT_SHIFT);
    {
        const sfpi::vFloat reciprocal_up =
            sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(sfpi::abs(sfpi::vConstFloatPrgm1)) + static_cast<std::uint32_t>(RECIPROCAL_ULP_BIAS));
        scalar_modulo_fast_bounded_stage<DIVISOR_HIGH_MANTISSA, 0u>(
            residual, sfpi::abs(sfpi::vConstFloatPrgm0), sfpi::addexp(reciprocal_up, -scalar_modulo_fast_bounded_split));
    }
    {
        const sfpi::vFloat reciprocal_up =
            sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(sfpi::abs(sfpi::vConstFloatPrgm1)) + static_cast<std::uint32_t>(RECIPROCAL_ULP_BIAS));
        scalar_modulo_fast_bounded_stage<DIVISOR_HIGH_MANTISSA, 0u>(
            residual, sfpi::addexp(sfpi::abs(sfpi::vConstFloatPrgm0), -scalar_modulo_fast_bounded_split), reciprocal_up);
    }

    v_if (residual != 0.0f)
    {
        residual = sfpi::addexp(residual, scalar_modulo_fast_bounded_split);
    }
    v_endif;
    return residual;
}

template <
    ScalarModuloKind KIND,
    int DIVISOR_EXPONENT,
    bool DIVIDEND_NEGATIVE,
    bool DIVISOR_NEGATIVE,
    int INPUT_EXPONENT_SHIFT,
    unsigned DIVISOR_HIGH_MANTISSA,
    unsigned RECIPROCAL_ULP_BIAS,
    int ITERATIONS = 32>
inline void calculate_scalar_modulo_fast_bounded_exact()
{
    static_assert(DIVISOR_EXPONENT >= -126 && DIVISOR_EXPONENT <= 127);

#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat input_magnitude = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[0]));
        if constexpr (DIVISOR_EXPONENT > scalar_modulo_stationary_working_exponent)
        {
            sfpi::vUInt output_bits = 0u;
            v_if (input_magnitude < sfpi::abs(sfpi::vConstFloatPrgm2))
            {
                output_bits = finalize_high_divisor_bypass_magnitude<KIND, (DIVIDEND_NEGATIVE != DIVISOR_NEGATIVE)>(input_magnitude);
            }
            v_else
            {
                sfpi::vFloat normalized_r;
                {
                    normalized_r = reduce_scalar_modulo_fast_bounded_exact<INPUT_EXPONENT_SHIFT, DIVISOR_HIGH_MANTISSA, RECIPROCAL_ULP_BIAS>(input_magnitude);
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
            sfpi::vFloat normalized_r = 0.0f;
            v_if (input_magnitude != 0.0f)
            {
                normalized_r = sfpi::setexp(
                    input_magnitude, sfpi::exexp(input_magnitude, sfpi::ExponentMode::Biased) + (scalar_modulo_stationary_working_exponent - DIVISOR_EXPONENT));
            }
            v_endif;

            v_if (input_magnitude >= sfpi::abs(sfpi::vConstFloatPrgm2))
            {
                normalized_r = reduce_scalar_modulo_fast_bounded_exact<INPUT_EXPONENT_SHIFT, DIVISOR_HIGH_MANTISSA, RECIPROCAL_ULP_BIAS>(input_magnitude);
            }
            v_endif;
            sfpi::dst_reg[0] = finalize_stationary_result<KIND, DIVISOR_EXPONENT, DIVIDEND_NEGATIVE, DIVISOR_NEGATIVE>(normalized_r);
        }
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
