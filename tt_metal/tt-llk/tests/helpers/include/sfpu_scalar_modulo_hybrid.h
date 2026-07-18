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
            v_if (value < 0.0f && reduced != 0.0f)
            {
                result = divisor - reduced;
            }
            v_endif;
            result = sfpi::copysgn(result, scalar);
        }
        else
        {
            result = sfpi::copysgn(reduced, value);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Diagnostic layout, one vector result per full-tile SFPU iteration:
//   0..8: a, b, reciprocal, scaled, q_hat, pre-correction residual,
//         safe predicate, fast residual, initial scaled divisor
//   9..28: fallback residual after steps 0..19
//   29: fallback residual after step 23
//   30: final fallback magnitude
//   31: final positive-divisor fmod result
template <int ITERATIONS = 32>
inline void calculate_scalar_modulo_diagnostic()
{
    const sfpi::vFloat divisor    = sfpi::abs(sfpi::vConstFloatPrgm0);
    const sfpi::vFloat reciprocal = sfpi::abs(sfpi::vConstFloatPrgm1);

#pragma GCC unroll 0
    for (int iteration = 0; iteration < ITERATIONS; ++iteration)
    {
        const sfpi::vFloat value           = sfpi::dst_reg[0];
        const sfpi::vFloat magnitude       = sfpi::abs(value);
        const sfpi::vFloat scaled          = magnitude * reciprocal;
        const sfpi::vFloat quotient        = scalar_modulo_truncate_positive(scaled);
        const sfpi::vFloat residual_before = magnitude - quotient * divisor;
        const sfpi::vFloat fast_residual   = scalar_modulo_fast_magnitude(magnitude, divisor, quotient);

        sfpi::vFloat safe = 0.0f;
        v_if (sfpi::exexp(scaled) < 22)
        {
            safe = 1.0f;
        }
        v_endif;

        sfpi::vFloat fallback       = magnitude;
        sfpi::vInt remaining        = sfpi::exexp(magnitude) - sfpi::exexp(divisor);
        sfpi::vFloat scaled_divisor = sfpi::setexp(divisor, sfpi::exexp(magnitude, sfpi::ExponentMode::Biased));
        v_if (scaled_divisor > magnitude)
        {
            scaled_divisor = sfpi::addexp(scaled_divisor, -1);
            remaining      = remaining - 1;
        }
        v_endif;
        const sfpi::vFloat initial_scaled_divisor = scaled_divisor;
        sfpi::vFloat checkpoint                   = fallback;

        int selected_step = iteration - 9;
        if (iteration == 29)
        {
            selected_step = 23;
        }

#pragma GCC unroll 0
        for (int step = 0; step < 254; ++step)
        {
            v_if (remaining >= 0)
            {
                v_if (fallback >= scaled_divisor)
                {
                    fallback = fallback - scaled_divisor;
                }
                v_endif;
                scaled_divisor = sfpi::addexp(scaled_divisor, -1);
                remaining      = remaining - 1;
            }
            v_endif;

            if (step == selected_step)
            {
                checkpoint = fallback;
            }
        }

        sfpi::vFloat output = fallback;
        if (iteration == 0)
        {
            output = magnitude;
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
            output = scaled;
        }
        else if (iteration == 4)
        {
            output = quotient;
        }
        else if (iteration == 5)
        {
            output = residual_before;
        }
        else if (iteration == 6)
        {
            output = safe;
        }
        else if (iteration == 7)
        {
            output = fast_residual;
        }
        else if (iteration == 8)
        {
            output = initial_scaled_divisor;
        }
        else if (iteration < 30)
        {
            output = checkpoint;
        }
        else if (iteration == 31)
        {
            output = sfpi::copysgn(fallback, value);
        }

        sfpi::dst_reg[0] = output;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
