// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu
{

inline void init_scalar_modulo_research(const std::uint32_t divisor, const std::uint32_t reciprocal)
{
    sfpi::vConstFloatPrgm0 = Converter::as_float(divisor);
    sfpi::vConstFloatPrgm1 = Converter::as_float(reciprocal);
}

inline sfpi::vFloat scalar_modulo_truncate_positive(const sfpi::vFloat& scaled)
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

inline sfpi::vFloat scalar_modulo_fast_magnitude(const sfpi::vFloat& magnitude, const sfpi::vFloat& divisor, const sfpi::vFloat& quotient)
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

inline sfpi::vFloat scalar_modulo_binary_reduce(const sfpi::vFloat& magnitude, const sfpi::vFloat& divisor)
{
    sfpi::vFloat residual       = magnitude;
    sfpi::vInt remaining        = sfpi::exexp(magnitude) - sfpi::exexp(divisor);
    sfpi::vFloat scaled_divisor = sfpi::setexp(divisor, sfpi::exexp(magnitude, sfpi::ExponentMode::Biased));

    v_if (scaled_divisor > magnitude)
    {
        scaled_divisor = sfpi::addexp(scaled_divisor, -1);
        remaining      = remaining - 1;
    }
    v_endif;

#pragma GCC unroll 0
    for (int step = 0; step < 254; ++step)
    {
        v_if (remaining >= 0)
        {
            v_if (residual >= scaled_divisor)
            {
                residual = residual - scaled_divisor;
            }
            v_endif;
            scaled_divisor = sfpi::addexp(scaled_divisor, -1);
            remaining      = remaining - 1;
        }
        v_endif;
    }
    return residual;
}

template <bool FLOOR_REMAINDER, int ITERATIONS = 32>
inline void calculate_scalar_modulo_hybrid()
{
    const sfpi::vFloat scalar     = sfpi::vConstFloatPrgm0;
    const sfpi::vFloat divisor    = sfpi::abs(scalar);
    const sfpi::vFloat reciprocal = sfpi::abs(sfpi::vConstFloatPrgm1);

#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat value     = sfpi::dst_reg[0];
        const sfpi::vFloat magnitude = sfpi::abs(value);
        const sfpi::vFloat scaled    = magnitude * reciprocal;
        const sfpi::vFloat quotient  = scalar_modulo_truncate_positive(scaled);
        sfpi::vFloat reduced         = scalar_modulo_fast_magnitude(magnitude, divisor, quotient);

        const sfpi::vInt scaled_exponent = sfpi::exexp(scaled);
        v_if (scaled_exponent >= 22)
        {
            reduced = scalar_modulo_binary_reduce(magnitude, divisor);
        }
        v_endif;

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

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Minimal diagnostic layout, one vector result per full-tile SFPU iteration:
//   0: a
//   1: b
//   2: reciprocal
//   3: scaled
//   4: q_hat
//   5: pre-correction residual
//   6: residual after the positive correction
//   7: residual after the negative correction
//   8: safe predicate
//   9..31: repeat the final corrected residual
//
// Compute each result in a separate scalar-control-flow arm. Keeping all of
// the intermediate vectors and fallback checkpoints live at once exceeds the
// SFPU local-register file and makes the compiler attempt an illegal spill.
template <int ITERATIONS = 32>
inline void calculate_scalar_modulo_diagnostic()
{
    const sfpi::vFloat divisor    = sfpi::abs(sfpi::vConstFloatPrgm0);
    const sfpi::vFloat reciprocal = sfpi::abs(sfpi::vConstFloatPrgm1);

#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vFloat output;
        if (iteration == 0)
        {
            output = sfpi::abs(value);
        }
        else if (iteration == 1)
        {
            output = divisor;
        }
        else if (iteration == 2)
        {
            output = reciprocal;
        }
        else if (iteration == 3)
        {
            output = sfpi::abs(value) * reciprocal;
        }
        else if (iteration == 4)
        {
            output = scalar_modulo_truncate_positive(sfpi::abs(value) * reciprocal);
        }
        else if (iteration == 5)
        {
            const sfpi::vFloat magnitude = sfpi::abs(value);
            const sfpi::vFloat quotient  = scalar_modulo_truncate_positive(magnitude * reciprocal);
            output                       = magnitude - quotient * divisor;
        }
        else if (iteration == 6)
        {
            const sfpi::vFloat magnitude = sfpi::abs(value);
            const sfpi::vFloat quotient  = scalar_modulo_truncate_positive(magnitude * reciprocal);
            output                       = magnitude - quotient * divisor;
            v_if (output >= divisor)
            {
                output = output - divisor;
            }
            v_endif;
        }
        else if (iteration == 8)
        {
            output = 0.0f;
            v_if (sfpi::exexp(sfpi::abs(value) * reciprocal) < 22)
            {
                output = 1.0f;
            }
            v_endif;
        }
        else
        {
            const sfpi::vFloat magnitude = sfpi::abs(value);
            const sfpi::vFloat quotient  = scalar_modulo_truncate_positive(magnitude * reciprocal);
            output                       = scalar_modulo_fast_magnitude(magnitude, divisor, quotient);
        }

        sfpi::dst_reg[0] = output;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
