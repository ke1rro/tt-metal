// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu
{

constexpr int scalar_modulo_chunk_bits       = 16;
constexpr int scalar_modulo_chunk_stages     = 17;
constexpr int scalar_modulo_component_bits   = 12;
constexpr int scalar_modulo_sfpmad_precision = 28;

inline void init_scalar_modulo_chunked_research(const std::uint32_t divisor, const std::uint32_t reciprocal, const std::uint32_t divisor_high)
{
    sfpi::vConstFloatPrgm0 = Converter::as_float(divisor);
    sfpi::vConstFloatPrgm1 = Converter::as_float(reciprocal);
    sfpi::vConstFloatPrgm2 = Converter::as_float(divisor_high);
}

inline sfpi::vFloat scalar_modulo_chunked_truncate_positive(const sfpi::vFloat& scaled)
{
    sfpi::vFloat quotient;
    const sfpi::vInt exponent = sfpi::exexp(scaled);

    v_if (exponent < 0)
    {
        quotient = 0.0f;
    }
    v_elseif (exponent < 23)
    {
        quotient = sfpi::as<sfpi::vFloat>(shft(shft(sfpi::as<sfpi::vUInt>(scaled), exponent - 23), 23 - exponent));
    }
    v_else
    {
        quotient = scaled;
    }
    v_endif;

    v_if (quotient > scaled)
    {
        quotient = quotient - 1.0f;
    }
    v_endif;
    return quotient;
}

inline sfpi::vFloat scalar_modulo_chunked_fast_magnitude(const sfpi::vFloat& magnitude, const sfpi::vFloat& divisor, const sfpi::vFloat& quotient)
{
    sfpi::vFloat residual = magnitude - quotient * divisor;
    v_if (residual >= divisor)
    {
        residual = residual - divisor;
    }
    v_endif;
    v_if (residual < 0.0f)
    {
        residual = residual + divisor;
    }
    v_endif;
    return residual;
}

// Proof-domain contract:
//   * residual/divisor and all active intermediates are positive normal FP32;
//   * reciprocal_up is nextUp(nextUp(RN32(1/divisor)));
//   * divisor_high contains the top 12 significand bits and divisor_low the
//     bottom 12, so each 16b-by-12b partial product fits the documented 28-bit
//     SFPMAD product precision;
//   * special values, reciprocal/product overflow, and FTZ/subnormal output
//     policy are excluded from this research prototype.
inline void scalar_modulo_chunked_reduce_inplace(sfpi::vFloat& residual)
{
#pragma GCC unroll 0
    for (int stage = 0; stage < scalar_modulo_chunk_stages; ++stage)
    {
        v_if (residual >= sfpi::abs(sfpi::vConstFloatPrgm0))
        {
            sfpi::vInt shift = sfpi::exexp(residual) - sfpi::exexp(sfpi::abs(sfpi::vConstFloatPrgm0)) - (scalar_modulo_chunk_bits - 1);
            v_if (shift < 0)
            {
                shift = 0;
            }
            v_endif;

            {
                const sfpi::vFloat quotient = scalar_modulo_chunked_truncate_positive(
                    sfpi::setexp(residual, sfpi::exexp(residual, sfpi::ExponentMode::Biased) - shift) *
                    sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(sfpi::vConstFloatPrgm1) + static_cast<std::uint32_t>(2)));

                residual = residual - quotient * sfpi::setexp(sfpi::vConstFloatPrgm2, sfpi::exexp(sfpi::vConstFloatPrgm2, sfpi::ExponentMode::Biased) + shift);

                sfpi::vFloat divisor_low = sfpi::abs(sfpi::vConstFloatPrgm0) - sfpi::vConstFloatPrgm2;
                v_if (divisor_low != 0.0f)
                {
                    divisor_low = sfpi::setexp(divisor_low, sfpi::exexp(divisor_low, sfpi::ExponentMode::Biased) + shift);
                }
                v_endif;
                residual = residual - quotient * divisor_low;
            }

            v_if (residual < 0.0f)
            {
                residual = residual +
                           sfpi::setexp(sfpi::abs(sfpi::vConstFloatPrgm0), sfpi::exexp(sfpi::abs(sfpi::vConstFloatPrgm0), sfpi::ExponentMode::Biased) + shift);
            }
            v_endif;
        }
        v_endif;
    }
}

template <bool FLOOR_REMAINDER>
inline sfpi::vFloat scalar_modulo_chunked_restore_sign(
    const sfpi::vFloat& value, const sfpi::vFloat& scalar, const sfpi::vFloat& divisor, const sfpi::vFloat& reduced)
{
    sfpi::vFloat result;
    if constexpr (FLOOR_REMAINDER)
    {
        result = reduced;
        v_if (((value < 0.0f && scalar >= 0.0f) || (value >= 0.0f && scalar < 0.0f)) && reduced != 0.0f)
        {
            result = divisor - reduced;
        }
        v_endif;
        result = sfpi::copysgn(result, scalar);
        v_if (reduced == 0.0f)
        {
            result = sfpi::copysgn(reduced, value);
        }
        v_endif;
    }
    else
    {
        result = sfpi::copysgn(reduced, value);
    }
    return result;
}

template <bool FLOOR_REMAINDER, int ITERATIONS = 32>
inline void calculate_scalar_modulo_chunked_hybrid()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        sfpi::vInt scaled_exponent;
        {
            const sfpi::vFloat classifier_input = sfpi::dst_reg[0];
            scaled_exponent                     = sfpi::exexp(sfpi::abs(classifier_input) * sfpi::vConstFloatPrgm1);
        }

        v_if (scaled_exponent < 22)
        {
            const sfpi::vFloat value     = sfpi::dst_reg[0];
            const sfpi::vFloat magnitude = sfpi::abs(value);
            const sfpi::vFloat scaled    = magnitude * sfpi::vConstFloatPrgm1;
            const sfpi::vFloat quotient  = scalar_modulo_chunked_truncate_positive(scaled);
            const sfpi::vFloat reduced   = scalar_modulo_chunked_fast_magnitude(magnitude, sfpi::abs(sfpi::vConstFloatPrgm0), quotient);
            sfpi::dst_reg[0] = scalar_modulo_chunked_restore_sign<FLOOR_REMAINDER>(value, sfpi::vConstFloatPrgm0, sfpi::abs(sfpi::vConstFloatPrgm0), reduced);
        }
        v_else
        {
            const sfpi::vFloat value = sfpi::dst_reg[0];
            sfpi::vFloat reduced     = sfpi::abs(value);
            scalar_modulo_chunked_reduce_inplace(reduced);
            sfpi::dst_reg[0] = scalar_modulo_chunked_restore_sign<FLOOR_REMAINDER>(value, sfpi::vConstFloatPrgm0, sfpi::abs(sfpi::vConstFloatPrgm0), reduced);
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

// Standalone robust form used as the performance lower bound for the chunked
// architecture.  If this form fails the performance gate, adding the cached
// fast path and its classifier cannot rescue the single-kernel design.
template <bool FLOOR_REMAINDER, int ITERATIONS = 32>
inline void calculate_scalar_modulo_chunked_robust()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        sfpi::vFloat reduced;
        {
            const sfpi::vFloat reduction_input = sfpi::dst_reg[0];
            reduced                            = sfpi::abs(reduction_input);
            scalar_modulo_chunked_reduce_inplace(reduced);
        }
        {
            const sfpi::vFloat value = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = scalar_modulo_chunked_restore_sign<FLOOR_REMAINDER>(value, sfpi::vConstFloatPrgm0, sfpi::abs(sfpi::vConstFloatPrgm0), reduced);
        }
        sfpi::dst_reg++;
    }
}

template <bool FLOOR_REMAINDER, int UNSAFE_LANES, int ITERATIONS = 32>
inline void calculate_scalar_modulo_chunked_robust_perf()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        {
            sfpi::vFloat value = 1.0f;
            v_if (sfpi::vConstTileId < UNSAFE_LANES * 2)
            {
                value = Converter::as_float(0x4c400002u);
            }
            v_endif;
            sfpi::dst_reg[0] = value;
        }

        sfpi::vFloat reduced;
        {
            const sfpi::vFloat reduction_input = sfpi::dst_reg[0];
            reduced                            = sfpi::abs(reduction_input);
            scalar_modulo_chunked_reduce_inplace(reduced);
        }
        {
            const sfpi::vFloat value = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = scalar_modulo_chunked_restore_sign<FLOOR_REMAINDER>(value, sfpi::vConstFloatPrgm0, sfpi::abs(sfpi::vConstFloatPrgm0), reduced);
        }
        sfpi::dst_reg++;
    }
}

template <bool FLOOR_REMAINDER, int UNSAFE_LANES, int ITERATIONS = 32>
inline void calculate_scalar_modulo_chunked_perf()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        {
            sfpi::vFloat value = 1.0f;
            v_if (sfpi::vConstTileId < UNSAFE_LANES * 2)
            {
                value = Converter::as_float(0x4c400002u);
            }
            v_endif;
            sfpi::dst_reg[0] = value;
        }

        sfpi::vInt scaled_exponent;
        {
            const sfpi::vFloat classifier_input = sfpi::dst_reg[0];
            scaled_exponent                     = sfpi::exexp(sfpi::abs(classifier_input) * sfpi::vConstFloatPrgm1);
        }
        v_if (scaled_exponent < 22)
        {
            const sfpi::vFloat value     = sfpi::dst_reg[0];
            const sfpi::vFloat magnitude = sfpi::abs(value);
            const sfpi::vFloat scaled    = magnitude * sfpi::vConstFloatPrgm1;
            const sfpi::vFloat quotient  = scalar_modulo_chunked_truncate_positive(scaled);
            const sfpi::vFloat reduced   = scalar_modulo_chunked_fast_magnitude(magnitude, sfpi::abs(sfpi::vConstFloatPrgm0), quotient);
            sfpi::dst_reg[0] = scalar_modulo_chunked_restore_sign<FLOOR_REMAINDER>(value, sfpi::vConstFloatPrgm0, sfpi::abs(sfpi::vConstFloatPrgm0), reduced);
        }
        v_else
        {
            const sfpi::vFloat value = sfpi::dst_reg[0];
            sfpi::vFloat reduced     = sfpi::abs(value);
            scalar_modulo_chunked_reduce_inplace(reduced);
            sfpi::dst_reg[0] = scalar_modulo_chunked_restore_sign<FLOOR_REMAINDER>(value, sfpi::vConstFloatPrgm0, sfpi::abs(sfpi::vConstFloatPrgm0), reduced);
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

// Branch-isolated raw-bit diagnostic layout:
//   0 a, 1 b, 2 RN reciprocal, 3 upward reciprocal, 4 scaled, 5 q_hat,
//   6 pre-correction residual, 7 after positive correction,
//   8 after negative correction, 9 chunked magnitude, 10 safe flag,
//   11 high divisor component, 12 low divisor component, 13..31 zero.
template <int ITERATIONS = 32>
inline void calculate_scalar_modulo_chunked_diagnostic()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat input = sfpi::dst_reg[0];
        const sfpi::vFloat value = sfpi::abs(input);
        sfpi::vFloat output      = 0.0f;
        if (iteration == 0)
        {
            output = value;
        }
        else if (iteration == 1)
        {
            output = sfpi::abs(sfpi::vConstFloatPrgm0);
        }
        else if (iteration == 2)
        {
            output = sfpi::vConstFloatPrgm1;
        }
        else if (iteration == 3)
        {
            output = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(sfpi::vConstFloatPrgm1) + static_cast<std::uint32_t>(2));
        }
        else if (iteration == 4)
        {
            output = value * sfpi::vConstFloatPrgm1;
        }
        else if (iteration == 5)
        {
            output = scalar_modulo_chunked_truncate_positive(value * sfpi::vConstFloatPrgm1);
        }
        else if (iteration == 6)
        {
            const sfpi::vFloat divisor  = sfpi::abs(sfpi::vConstFloatPrgm0);
            const sfpi::vFloat quotient = scalar_modulo_chunked_truncate_positive(value * sfpi::vConstFloatPrgm1);
            output                      = value - quotient * divisor;
        }
        else if (iteration == 7)
        {
            const sfpi::vFloat divisor  = sfpi::abs(sfpi::vConstFloatPrgm0);
            const sfpi::vFloat quotient = scalar_modulo_chunked_truncate_positive(value * sfpi::vConstFloatPrgm1);
            output                      = value - quotient * divisor;
            v_if (output >= divisor)
            {
                output = output - divisor;
            }
            v_endif;
        }
        else if (iteration == 8)
        {
            const sfpi::vFloat divisor  = sfpi::abs(sfpi::vConstFloatPrgm0);
            const sfpi::vFloat quotient = scalar_modulo_chunked_truncate_positive(value * sfpi::vConstFloatPrgm1);
            output                      = scalar_modulo_chunked_fast_magnitude(value, divisor, quotient);
        }
        else if (iteration == 9)
        {
            output = value;
            scalar_modulo_chunked_reduce_inplace(output);
        }
        else if (iteration == 10)
        {
            v_if (sfpi::exexp(value * sfpi::vConstFloatPrgm1) < 22)
            {
                output = 1.0f;
            }
            v_endif;
        }
        else if (iteration == 11)
        {
            output = sfpi::vConstFloatPrgm2;
        }
        else if (iteration == 12)
        {
            output = sfpi::abs(sfpi::vConstFloatPrgm0) - sfpi::vConstFloatPrgm2;
        }

        sfpi::dst_reg[0] = output;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
