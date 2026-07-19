// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu
{

constexpr int scalar_modulo_fixed_chunk_bits = 16;
constexpr int scalar_modulo_fixed_step       = scalar_modulo_fixed_chunk_bits - 1;

inline void init_scalar_modulo_fixed_schedule_research(const std::uint32_t divisor, const std::uint32_t initial_inverse)
{
    sfpi::vConstFloatPrgm0 = Converter::as_float(divisor);
    sfpi::vConstFloatPrgm1 = Converter::as_float(initial_inverse);
}

// Proof-domain contract:
//   * the input is zero or finite normal FP32; the divisor, scaled divisor,
//     inverse, and active intermediates are finite normal FP32 values;
//   * initial_inverse is nextUp(nextUp(RN32(1/abs(divisor)))) *
//     2**(-START_SHIFT), except that an exact power-of-two reciprocal is not
//     biased;
//   * START_SHIFT=max(112-unbiased_exponent(abs(divisor)), 0), so every local
//     quotient is below 2**16;
//   * DIVISOR_HIGH_MANTISSA is the top 12 significand bits encoded as the
//     SFPSETMAN immediate (its low bit is zero).  The resulting 16b-by-12b
//     products fit the documented 28-bit SFPMAD product precision;
//   * FTZ/subnormal results, special values, and reciprocal/product overflow
//     outside the exponent-127 case are outside this test-only prototype.  The
//     public wrapper below closes the observed maximum-FP32 partial-product
//     corner with an exact a/2 pre-reduction.
template <unsigned DIVISOR_HIGH_MANTISSA>
inline void scalar_modulo_fixed_stage(sfpi::vFloat& residual, const sfpi::vFloat& scaled_divisor, const sfpi::vFloat& inverse_scaled_divisor)
{
    v_if (residual >= scaled_divisor)
    {
        const sfpi::vFloat scaled_quotient   = residual * inverse_scaled_divisor;
        const sfpi::vUInt16 quotient_integer = sfpi::convert<sfpi::vUInt16>(scaled_quotient, sfpi::RoundMode::Nearest);
        const sfpi::vFloat quotient          = sfpi::convert<sfpi::vFloat>(quotient_integer, sfpi::RoundMode::Nearest);

        const sfpi::vFloat divisor_high = sfpi::setman(scaled_divisor, DIVISOR_HIGH_MANTISSA);
        const sfpi::vFloat divisor_low  = scaled_divisor - divisor_high;
        residual                        = residual - quotient * divisor_high;
        residual                        = residual - quotient * divisor_low;

        // The upward reciprocal can overestimate floor(residual/divisor) by
        // exactly one; the correction restores the exact nonnegative residue.
        v_if (residual < 0.0f)
        {
            residual = residual + scaled_divisor;
        }
        v_endif;
    }
    v_endif;
}

template <int START_SHIFT, unsigned DIVISOR_HIGH_MANTISSA>
inline void scalar_modulo_fixed_reduce_inplace(sfpi::vFloat& residual)
{
    static_assert(START_SHIFT >= 0 && START_SHIFT <= 238);
    static_assert(DIVISOR_HIGH_MANTISSA <= 0xFFFu);
    static_assert((DIVISOR_HIGH_MANTISSA & 1u) == 0u);

    constexpr int full_decrements = START_SHIFT / scalar_modulo_fixed_step;
    constexpr int final_decrement = START_SHIFT % scalar_modulo_fixed_step;

    sfpi::vFloat scaled_divisor         = sfpi::addexp(sfpi::abs(sfpi::vConstFloatPrgm0), START_SHIFT);
    sfpi::vFloat inverse_scaled_divisor = sfpi::vConstFloatPrgm1;

#pragma GCC unroll 0
    for (int stage = 0; stage < full_decrements; ++stage)
    {
        scalar_modulo_fixed_stage<DIVISOR_HIGH_MANTISSA>(residual, scaled_divisor, inverse_scaled_divisor);
        scaled_divisor         = sfpi::addexp(scaled_divisor, -scalar_modulo_fixed_step);
        inverse_scaled_divisor = sfpi::addexp(inverse_scaled_divisor, scalar_modulo_fixed_step);
    }

    if constexpr (final_decrement != 0)
    {
        scalar_modulo_fixed_stage<DIVISOR_HIGH_MANTISSA>(residual, scaled_divisor, inverse_scaled_divisor);
        scaled_divisor         = sfpi::addexp(scaled_divisor, -final_decrement);
        inverse_scaled_divisor = sfpi::addexp(inverse_scaled_divisor, final_decrement);
    }

    scalar_modulo_fixed_stage<DIVISOR_HIGH_MANTISSA>(residual, scaled_divisor, inverse_scaled_divisor);
}

template <bool FLOOR_REMAINDER>
inline sfpi::vFloat scalar_modulo_fixed_restore_sign(
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

template <bool FLOOR_REMAINDER, int START_SHIFT, unsigned DIVISOR_HIGH_MANTISSA, int ITERATIONS = 32>
inline void calculate_scalar_modulo_fixed_schedule()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vFloat reduced     = sfpi::abs(value);

        // Exact top-range pre-reduction.  For exponent-127 magnitudes, h=a/2
        // is normal and exact.  After reducing h, a mod b is (2*(h mod b))
        // mod b, which needs at most one final subtraction.  This keeps the
        // first split partial product below the FP32 overflow range.
        v_if (sfpi::exexp(reduced) == 127)
        {
            reduced = sfpi::addexp(reduced, -1);
        }
        v_endif;

        scalar_modulo_fixed_reduce_inplace<START_SHIFT, DIVISOR_HIGH_MANTISSA>(reduced);

        v_if (sfpi::exexp(sfpi::abs(value)) == 127 && reduced != 0.0f)
        {
            reduced = sfpi::addexp(reduced, 1);
            v_if (reduced >= sfpi::abs(sfpi::vConstFloatPrgm0))
            {
                reduced = reduced - sfpi::abs(sfpi::vConstFloatPrgm0);
            }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = scalar_modulo_fixed_restore_sign<FLOOR_REMAINDER>(value, sfpi::vConstFloatPrgm0, sfpi::abs(sfpi::vConstFloatPrgm0), reduced);
        sfpi::dst_reg++;
    }
}

template <bool FLOOR_REMAINDER, int START_SHIFT, unsigned DIVISOR_HIGH_MANTISSA, int ACTIVE_LANES, int ITERATIONS = 32>
inline void calculate_scalar_modulo_fixed_schedule_perf()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        sfpi::vFloat value = 1.0f;
        v_if (sfpi::vConstTileId < ACTIVE_LANES * 2)
        {
            value = Converter::as_float(0x4c400002u);
        }
        v_endif;
        sfpi::dst_reg[0] = value;

        sfpi::vFloat reduced = sfpi::abs(value);
        v_if (sfpi::exexp(reduced) == 127)
        {
            reduced = sfpi::addexp(reduced, -1);
        }
        v_endif;
        scalar_modulo_fixed_reduce_inplace<START_SHIFT, DIVISOR_HIGH_MANTISSA>(reduced);
        v_if (sfpi::exexp(sfpi::abs(value)) == 127 && reduced != 0.0f)
        {
            reduced = sfpi::addexp(reduced, 1);
            v_if (reduced >= sfpi::abs(sfpi::vConstFloatPrgm0))
            {
                reduced = reduced - sfpi::abs(sfpi::vConstFloatPrgm0);
            }
            v_endif;
        }
        v_endif;
        sfpi::dst_reg[0] = scalar_modulo_fixed_restore_sign<FLOOR_REMAINDER>(value, sfpi::vConstFloatPrgm0, sfpi::abs(sfpi::vConstFloatPrgm0), reduced);
        sfpi::dst_reg++;
    }
}

// Raw-bit diagnostic layout for the final unscaled stage:
//   0 input magnitude, 1 divisor, 2 upward reciprocal, 3 scaled quotient,
//   4 rounded UINT16 quotient converted to FP32, 5 divisor high,
//   6 divisor low, 7 after high subtraction, 8 after low subtraction,
//   9 after negative correction, 10..31 zero.
template <int START_SHIFT, unsigned DIVISOR_HIGH_MANTISSA, int ITERATIONS = 32>
inline void calculate_scalar_modulo_fixed_schedule_diagnostic()
{
#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat register_input = sfpi::dst_reg[0];
        const sfpi::vFloat input          = sfpi::abs(register_input);
        sfpi::vFloat output               = 0.0f;
        if (iteration == 0)
        {
            output = input;
        }
        else if (iteration == 1)
        {
            output = sfpi::abs(sfpi::vConstFloatPrgm0);
        }
        else if (iteration == 2)
        {
            output = sfpi::addexp(sfpi::vConstFloatPrgm1, START_SHIFT);
        }
        else if (iteration == 3)
        {
            output = input * sfpi::addexp(sfpi::vConstFloatPrgm1, START_SHIFT);
        }
        else if (iteration == 4)
        {
            const sfpi::vFloat scaled_quotient   = input * sfpi::addexp(sfpi::vConstFloatPrgm1, START_SHIFT);
            const sfpi::vUInt16 quotient_integer = sfpi::convert<sfpi::vUInt16>(scaled_quotient, sfpi::RoundMode::Nearest);
            output                               = sfpi::convert<sfpi::vFloat>(quotient_integer, sfpi::RoundMode::Nearest);
        }
        else if (iteration == 5)
        {
            output = sfpi::setman(sfpi::abs(sfpi::vConstFloatPrgm0), DIVISOR_HIGH_MANTISSA);
        }
        else if (iteration == 6)
        {
            const sfpi::vFloat divisor = sfpi::abs(sfpi::vConstFloatPrgm0);
            output                     = divisor - sfpi::setman(divisor, DIVISOR_HIGH_MANTISSA);
        }
        else if (iteration == 7)
        {
            const sfpi::vFloat scaled_quotient   = input * sfpi::addexp(sfpi::vConstFloatPrgm1, START_SHIFT);
            const sfpi::vUInt16 quotient_integer = sfpi::convert<sfpi::vUInt16>(scaled_quotient, sfpi::RoundMode::Nearest);
            const sfpi::vFloat quotient          = sfpi::convert<sfpi::vFloat>(quotient_integer, sfpi::RoundMode::Nearest);
            const sfpi::vFloat divisor           = sfpi::abs(sfpi::vConstFloatPrgm0);
            output                               = input - quotient * sfpi::setman(divisor, DIVISOR_HIGH_MANTISSA);
        }
        else if (iteration == 8)
        {
            const sfpi::vFloat scaled_quotient   = input * sfpi::addexp(sfpi::vConstFloatPrgm1, START_SHIFT);
            const sfpi::vUInt16 quotient_integer = sfpi::convert<sfpi::vUInt16>(scaled_quotient, sfpi::RoundMode::Nearest);
            const sfpi::vFloat quotient          = sfpi::convert<sfpi::vFloat>(quotient_integer, sfpi::RoundMode::Nearest);
            const sfpi::vFloat divisor           = sfpi::abs(sfpi::vConstFloatPrgm0);
            const sfpi::vFloat divisor_high      = sfpi::setman(divisor, DIVISOR_HIGH_MANTISSA);
            output                               = input - quotient * divisor_high;
            output                               = output - quotient * (divisor - divisor_high);
        }
        else if (iteration == 9)
        {
            const sfpi::vFloat scaled_quotient   = input * sfpi::addexp(sfpi::vConstFloatPrgm1, START_SHIFT);
            const sfpi::vUInt16 quotient_integer = sfpi::convert<sfpi::vUInt16>(scaled_quotient, sfpi::RoundMode::Nearest);
            const sfpi::vFloat quotient          = sfpi::convert<sfpi::vFloat>(quotient_integer, sfpi::RoundMode::Nearest);
            const sfpi::vFloat divisor           = sfpi::abs(sfpi::vConstFloatPrgm0);
            const sfpi::vFloat divisor_high      = sfpi::setman(divisor, DIVISOR_HIGH_MANTISSA);
            output                               = input - quotient * divisor_high;
            output                               = output - quotient * (divisor - divisor_high);
            v_if (output < 0.0f)
            {
                output = output + divisor;
            }
            v_endif;
        }
        sfpi::dst_reg[0] = output;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
