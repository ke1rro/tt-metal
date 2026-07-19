#!/usr/bin/env python3
"""Exact host model for the test-only scalar modulo fixed schedule.

This is the Architecture-B robust speed experiment.  The scalar divisor lets
the host choose a fixed sequence of power-of-two-scaled divisors.  Each active
stage has a local quotient below 2**16, so the SFPI implementation can use the
portable FP32 -> UINT16 -> FP32 conversion pair instead of a general FP32
truncate sequence.

The model intentionally has the same finite-normal proof-domain exclusions as
the earlier C16 research.  It reports subnormal/FTZ and product-overflow cases
rather than treating them as successes.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
from sfpi_modulo_chunked_reference import (
    DEFAULT_DIVISORS,
    FP32,
    U32,
    DomainExclusion,
    boundary_cases,
    exact_mod_normal,
    fp32_bits,
    fp32_from_bits,
    is_normal_positive,
    normal_parts,
    reciprocal_up_two,
    scale_normal,
    verify_component_subtractions,
)

CHUNK_BITS = 16
CHUNK_STEP = CHUNK_BITS - 1
MAX_INPUT_EXPONENT = 127
# Keeping the first scaled divisor at exponent >=112 bounds the initial ratio
# strictly below 2**16, even for a maximum-exponent dividend.
INITIAL_DIVISOR_EXPONENT = MAX_INPUT_EXPONENT - CHUNK_STEP


@dataclass
class FixedScheduleMetrics:
    tested: int = 0
    failures: int = 0
    exclusions: int = 0
    max_schedule_stages: int = 0
    max_active_stages: int = 0
    min_quotient_error: int = 1 << 30
    max_quotient_error: int = -(1 << 30)
    max_local_quotient: int = 0
    prehalved_inputs: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)

    def record_exclusion(self, error: DomainExclusion) -> None:
        self.exclusions += 1
        reason = str(error)
        if reason.startswith("biased reciprocal is not normal"):
            reason = "biased reciprocal is not normal"
        self.exclusion_reasons[reason] = self.exclusion_reasons.get(reason, 0) + 1

    def summary(self) -> str:
        if self.min_quotient_error == 1 << 30:
            quotient_errors = "n/a"
        else:
            quotient_errors = f"[{self.min_quotient_error},{self.max_quotient_error}]"
        return (
            f"fixed_c16 tested={self.tested} failures={self.failures} exclusions={self.exclusions} "
            f"schedule_stages={self.max_schedule_stages} active_stages={self.max_active_stages} "
            f"q_error={quotient_errors} max_q={self.max_local_quotient} "
            f"prehalved={self.prehalved_inputs}"
        )


def fixed_schedule(divisor: np.float32) -> tuple[int, ...]:
    """Return the scalar shift sequence K0, K0-15, ..., 0."""

    _, _, divisor_exponent = normal_parts(divisor)
    shift = max(INITIAL_DIVISOR_EXPONENT - divisor_exponent, 0)
    shifts: list[int] = []
    while True:
        shifts.append(shift)
        if shift == 0:
            return tuple(shifts)
        shift = max(shift - CHUNK_STEP, 0)


def nearest_away_uint16(value: np.float32) -> int:
    """Model WH/BH SFPSTOCHRND FP32->UINT16 nearest-away plus saturation."""

    rounded = math.floor(float(value) + 0.5)
    return min(rounded, 65535)


def fixed_schedule_reduce(
    dividend: np.float32,
    divisor: np.float32,
    metrics: FixedScheduleMetrics | None = None,
    prehalve_max_exponent: bool = False,
) -> np.float32:
    if not (is_normal_positive(dividend) and is_normal_positive(divisor)):
        raise DomainExclusion("fixed_schedule_reduce requires positive normal inputs")

    reciprocal = reciprocal_up_two(divisor)
    shifts = fixed_schedule(divisor)
    _, _, dividend_exponent = normal_parts(dividend)
    prehalved = prehalve_max_exponent and dividend_exponent == MAX_INPUT_EXPONENT
    residual = scale_normal(dividend, -1) if prehalved else dividend
    active_stages = 0

    for shift in shifts:
        scaled_divisor = scale_normal(divisor, shift)
        inverse_scaled_divisor = scale_normal(reciprocal, -shift)
        if residual < scaled_divisor:
            continue

        with np.errstate(all="ignore"):
            scaled_quotient = FP32(residual * inverse_scaled_divisor)
        if not is_normal_positive(scaled_quotient):
            raise DomainExclusion("local reciprocal product is not positive normal")

        quotient_hat = nearest_away_uint16(scaled_quotient)
        quotient_exact, next_residual = exact_mod_normal(residual, scaled_divisor)
        quotient_error = quotient_hat - quotient_exact
        if quotient_error not in (0, 1):
            raise AssertionError(
                f"local quotient error {quotient_error} is not one-sided: "
                f"a=0x{fp32_bits(dividend):08x} b=0x{fp32_bits(divisor):08x} "
                f"shift={shift} z=0x{fp32_bits(scaled_quotient):08x} "
                f"q_hat={quotient_hat} q_exact={quotient_exact}"
            )

        verify_component_subtractions(
            residual,
            scaled_divisor,
            quotient_hat,
            quotient_exact,
            CHUNK_BITS,
        )
        residual = next_residual
        active_stages += 1

        if metrics is not None:
            metrics.min_quotient_error = min(metrics.min_quotient_error, quotient_error)
            metrics.max_quotient_error = max(metrics.max_quotient_error, quotient_error)
            metrics.max_local_quotient = max(metrics.max_local_quotient, quotient_hat)

    if prehalved and residual:
        residual = scale_normal(residual, 1)
        if residual >= divisor:
            quotient, residual = exact_mod_normal(residual, divisor)
            if quotient != 1:
                raise AssertionError(f"post-double correction quotient is {quotient}, expected 1")

    _, expected = exact_mod_normal(dividend, divisor)
    if fp32_bits(residual) != fp32_bits(expected):
        raise AssertionError(
            f"final mismatch a=0x{fp32_bits(dividend):08x} b=0x{fp32_bits(divisor):08x}: "
            f"got=0x{fp32_bits(residual):08x} expected=0x{fp32_bits(expected):08x}"
        )

    if metrics is not None:
        metrics.tested += 1
        metrics.max_schedule_stages = max(metrics.max_schedule_stages, len(shifts))
        metrics.max_active_stages = max(metrics.max_active_stages, active_stages)
        metrics.prehalved_inputs += int(prehalved)
    return residual


def evaluate(
    dividend: np.float32,
    divisor: np.float32,
    metrics: FixedScheduleMetrics,
    prehalve_max_exponent: bool,
) -> None:
    try:
        fixed_schedule_reduce(dividend, divisor, metrics, prehalve_max_exponent)
    except DomainExclusion as error:
        metrics.record_exclusion(error)
    except AssertionError:
        metrics.failures += 1
        raise


def run_deterministic(metrics: FixedScheduleMetrics, prehalve_max_exponent: bool) -> None:
    cases = boundary_cases(DEFAULT_DIVISORS)
    for dividend, divisor in cases:
        evaluate(dividend, divisor, metrics, prehalve_max_exponent)
    print(f"deterministic_and_boundaries={len(cases)}")


def run_bf16(metrics: FixedScheduleMetrics, prehalve_max_exponent: bool) -> None:
    bf16_bits = np.arange(1, 0x7F80, dtype=U32)
    values = (bf16_bits << 16).view(FP32)
    pairs = 0
    for divisor in DEFAULT_DIVISORS:
        for dividend in values:
            if is_normal_positive(dividend):
                evaluate(dividend, divisor, metrics, prehalve_max_exponent)
                pairs += 1
    print(f"bf16_positive_finite_pairs={pairs}")


def run_random(
    metrics: FixedScheduleMetrics,
    count: int,
    seed: int,
    random_divisors: bool,
    prehalve_max_exponent: bool,
) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(count):
        dividend = fp32_from_bits(int(rng.integers(0x00800000, 0x7F7FFFFF, dtype=U32)))
        if random_divisors:
            divisor = fp32_from_bits(int(rng.integers(0x00800000, 0x7F7FFFFF, dtype=U32)))
        else:
            divisor = DEFAULT_DIVISORS[int(rng.integers(0, len(DEFAULT_DIVISORS)))]
        evaluate(dividend, divisor, metrics, prehalve_max_exponent)
    divisor_mode = "random_normal" if random_divisors else "selected"
    print(f"random_normal_pairs={count} divisor_mode={divisor_mode} seed=0x{seed:x}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=100_000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x5F91)
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--random-divisors", action="store_true")
    parser.add_argument("--prehalve-max-exponent", action="store_true")
    args = parser.parse_args()

    metrics = FixedScheduleMetrics()
    run_deterministic(metrics, args.prehalve_max_exponent)
    if not args.skip_bf16:
        run_bf16(metrics, args.prehalve_max_exponent)
    run_random(metrics, args.random, args.seed, args.random_divisors, args.prehalve_max_exponent)
    print(metrics.summary())
    for reason, count in sorted(metrics.exclusion_reasons.items()):
        print(f"exclusion: {reason}={count}")
    if metrics.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
