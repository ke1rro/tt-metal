#!/usr/bin/env python3
"""Proof-oriented host research for fixed-stage SFPI modulo reducers.

The candidate reduces a positive normal binary32 residual in radix ``2**C``.
For non-power-of-two divisors it deliberately uses a reciprocal rounded two
FP32 ULPs upward.  Under normal RNE multiplication this makes the local
quotient estimate one-sided (power-of-two division is handled exactly):

    floor(r / d) <= q_hat <= floor(r / d) + 1.

The overestimate is important.  A quotient underestimate can form a value in
``[d, 2*d)`` which is not necessarily representable before correction.  An
overestimate forms a value in ``(-d, 0]``; its magnitude and the corrected
remainder are representable on the divisor's binary lattice.

Tensix SFPMAD retains four product bits beyond FP32 rather than a complete
infinite-precision product.  Therefore a C-bit quotient cannot be multiplied
directly by a general 24-bit divisor.  The model splits the divisor significand
into groups of at most ``28-C`` bits.  Every partial product then has at most 28
significant bits, matching the documented Blackhole/Wormhole SFPMAD product
precision.  The model verifies that every intermediate subtraction is itself
representable as binary32 before accepting a stage.

This tool covers positive finite normal arithmetic.  It reports rather than
silently accepts cases requiring subnormal internal values, FTZ output policy,
reciprocal/product overflow, or non-finite handling.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
from sfpi_modulo_hybrid_reference import (
    FP32,
    U32,
    deterministic_cases,
    fp32_bits,
    fp32_from_bits,
    is_normal_positive,
    is_power_of_two_normal,
)

CHUNK_BITS_VALUES = (8, 12, 16, 18, 20)
SFPMAD_PRODUCT_BITS = 28
DEFAULT_DIVISORS = tuple(
    FP32(value)
    for value in (
        np.finfo(FP32).tiny,
        2.0**-64,
        0.5,
        2.0,
        3.0,
        5.0,
        7.0,
        8.0,
        10.0,
        0.1,
        0.3,
    )
)


class DomainExclusion(ValueError):
    """The normal-domain proof does not cover this pair or intermediate."""


@dataclass
class CandidateMetrics:
    chunk_bits: int
    tested: int = 0
    failures: int = 0
    exclusions: int = 0
    max_stages: int = 0
    min_progress: int = 1 << 30
    min_quotient_error: int = 1 << 30
    max_quotient_error: int = -(1 << 30)
    max_corrections: int = 0
    max_local_quotient: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def component_bits(self) -> int:
        return SFPMAD_PRODUCT_BITS - self.chunk_bits

    @property
    def component_count(self) -> int:
        return math.ceil(24 / self.component_bits)

    @property
    def fixed_stages(self) -> int:
        # The largest normal-FP32 exponent gap is 253.  A nonterminal stage
        # reduces it by at least CHUNK_BITS-1.
        return math.ceil(253 / (self.chunk_bits - 1))

    def record(self, result: ReductionResult) -> None:
        self.tested += 1
        self.max_stages = max(self.max_stages, result.stages)
        self.max_corrections = max(self.max_corrections, result.max_corrections)
        self.max_local_quotient = max(self.max_local_quotient, result.max_local_quotient)
        if result.min_progress is not None:
            self.min_progress = min(self.min_progress, result.min_progress)
        if result.min_quotient_error is not None:
            self.min_quotient_error = min(self.min_quotient_error, result.min_quotient_error)
            self.max_quotient_error = max(self.max_quotient_error, result.max_quotient_error)

    def record_exclusion(self, error: DomainExclusion) -> None:
        self.exclusions += 1
        reason = str(error)
        self.exclusion_reasons[reason] = self.exclusion_reasons.get(reason, 0) + 1

    def summary(self) -> str:
        min_progress = "n/a" if self.min_progress == 1 << 30 else str(self.min_progress)
        if self.min_quotient_error == 1 << 30:
            quotient_errors = "n/a"
        else:
            quotient_errors = f"[{self.min_quotient_error},{self.max_quotient_error}]"
        return (
            f"chunk_bits={self.chunk_bits:2d} fixed_stages={self.fixed_stages:2d} "
            f"observed_stages={self.max_stages:2d} components={self.component_count}x<={self.component_bits}b "
            f"q_error={quotient_errors} corrections={self.max_corrections} "
            f"min_progress={min_progress} tested={self.tested} "
            f"failures={self.failures} exclusions={self.exclusions}"
        )

    def exclusion_summary(self) -> str:
        if not self.exclusion_reasons:
            return ""
        details = ", ".join(f"{reason}={count}" for reason, count in sorted(self.exclusion_reasons.items()))
        return f"chunk_bits={self.chunk_bits:2d} exclusion_reasons: {details}"


@dataclass(frozen=True)
class ReductionResult:
    residual: np.float32
    stages: int
    min_progress: int | None
    min_quotient_error: int | None
    max_quotient_error: int | None
    max_corrections: int
    max_local_quotient: int


def normal_parts(value: np.float32) -> tuple[int, int, int]:
    """Return ``(24-bit significand, unit exponent, unbiased exponent)``."""

    bits = fp32_bits(value)
    biased = (bits >> 23) & 0xFF
    if bits >> 31 or not 0 < biased < 0xFF:
        raise DomainExclusion(f"not positive normal: 0x{bits:08x}")
    significand = (1 << 23) | (bits & 0x7FFFFF)
    unbiased = biased - 127
    return significand, unbiased - 23, unbiased


def precision_span(value: int) -> int:
    """Number of significant binary places after removing trailing zeroes."""

    if value == 0:
        return 0
    magnitude = abs(value)
    trailing = (magnitude & -magnitude).bit_length() - 1
    return magnitude.bit_length() - trailing


def scale_normal(value: np.float32, exponent_delta: int) -> np.float32:
    """Exact power-of-two scaling via exponent-field adjustment."""

    bits = fp32_bits(value)
    biased = (bits >> 23) & 0xFF
    scaled_biased = biased + exponent_delta
    if not 0 < scaled_biased < 0xFF:
        raise DomainExclusion(f"exponent scaling leaves normal range: value=0x{bits:08x} delta={exponent_delta}")
    return fp32_from_bits((bits & 0x807FFFFF) | (scaled_biased << 23))


def reciprocal_up_two(divisor: np.float32) -> np.float32:
    """Return ``nextUp(nextUp(RN32(1/divisor)))``."""

    with np.errstate(all="ignore"):
        reciprocal = FP32(FP32(1.0) / divisor)
    # Power-of-two division is exact.  Keeping its exact reciprocal avoids an
    # unnecessary overestimate and the corresponding product-overflow corner
    # at the top of the FP32 range.
    if not is_power_of_two_normal(divisor):
        reciprocal = np.nextafter(reciprocal, FP32(math.inf), dtype=FP32)
        reciprocal = np.nextafter(reciprocal, FP32(math.inf), dtype=FP32)
    if not is_normal_positive(reciprocal):
        raise DomainExclusion(f"biased reciprocal is not normal for divisor=0x{fp32_bits(divisor):08x}")
    return reciprocal


def compose_normal(significand: int, unit_exponent: int) -> np.float32:
    """Create an exact positive normal FP32 value from an integer lattice."""

    if significand == 0:
        return FP32(0.0)
    significant_bits = significand.bit_length()
    if significant_bits > 24:
        raise AssertionError(f"nonrepresentable significand with {significant_bits} bits")
    unbiased = unit_exponent + significant_bits - 1
    if unbiased < -126:
        raise DomainExclusion("exact intermediate/result is subnormal and subject to SFPU FTZ")
    if unbiased > 127:
        raise DomainExclusion("exact intermediate/result overflows FP32")
    mantissa = (significand << (24 - significant_bits)) & 0x7FFFFF
    return fp32_from_bits(((unbiased + 127) << 23) | mantissa)


def exact_mod_normal(dividend: np.float32, divisor: np.float32) -> tuple[int, np.float32]:
    """Exact quotient/remainder for positive normal FP32 operands."""

    dividend_m, dividend_unit, _ = normal_parts(dividend)
    divisor_m, divisor_unit, _ = normal_parts(divisor)
    common_unit = min(dividend_unit, divisor_unit)
    dividend_integer = dividend_m << (dividend_unit - common_unit)
    divisor_integer = divisor_m << (divisor_unit - common_unit)
    quotient, remainder = divmod(dividend_integer, divisor_integer)
    return quotient, compose_normal(remainder, common_unit)


def significand_components(significand: int, component_bits: int) -> tuple[int, ...]:
    """Split a 24-bit significand into high-to-low fixed-width components."""

    components: list[int] = []
    remaining = 24
    while remaining:
        width = min(component_bits, remaining)
        remaining -= width
        chunk = (significand >> remaining) & ((1 << width) - 1)
        if chunk:
            components.append(chunk << remaining)
    return tuple(components)


def verify_component_subtractions(
    residual: np.float32,
    scaled_divisor: np.float32,
    quotient_hat: int,
    quotient_exact: int,
    chunk_bits: int,
) -> None:
    """Verify the proof obligations imposed by partial SFPMAD fusion."""

    residual_m, residual_unit, _ = normal_parts(residual)
    divisor_m, divisor_unit, _ = normal_parts(scaled_divisor)
    if residual_unit < divisor_unit:
        raise AssertionError("scaled divisor must not have a coarser lattice than the residual")

    component_bits = SFPMAD_PRODUCT_BITS - chunk_bits
    residual_integer = residual_m << (residual_unit - divisor_unit)
    for component in significand_components(divisor_m, component_bits):
        component_unbiased = divisor_unit + component.bit_length() - 1
        if component_unbiased < -126:
            raise DomainExclusion("a split divisor component is subnormal and would be flushed")

        product = quotient_hat * component
        if precision_span(product) > SFPMAD_PRODUCT_BITS:
            raise AssertionError("partial product exceeds the documented 28-bit SFPMAD precision")
        if product and divisor_unit + product.bit_length() - 1 >= 128:
            raise DomainExclusion("a partial product reaches the SFPMAD overflow range")

        residual_integer -= product
        if precision_span(residual_integer) > 24:
            raise AssertionError("a component subtraction is not exactly representable as FP32")
        if residual_integer and divisor_unit + abs(residual_integer).bit_length() - 1 < -126:
            raise DomainExclusion("a component subtraction produces a subnormal intermediate")

    if quotient_hat == quotient_exact + 1:
        residual_integer += divisor_m
    elif quotient_hat != quotient_exact:
        raise AssertionError(f"local quotient error is not one-sided: {quotient_hat - quotient_exact}")

    _, exact_remainder = exact_mod_normal(residual, scaled_divisor)
    expected_m, expected_unit, _ = normal_parts(exact_remainder) if exact_remainder else (0, divisor_unit, 0)
    if expected_m:
        unit_delta = expected_unit - divisor_unit
        if unit_delta >= 0:
            expected_integer = expected_m << unit_delta
        else:
            discarded_mask = (1 << -unit_delta) - 1
            if expected_m & discarded_mask:
                raise AssertionError("exact remainder is not on the scaled-divisor lattice")
            expected_integer = expected_m >> -unit_delta
    else:
        expected_integer = 0
    if residual_integer != expected_integer:
        raise AssertionError("component reduction does not reconstruct the exact stage remainder")


def chunked_reduce(dividend: np.float32, divisor: np.float32, chunk_bits: int) -> ReductionResult:
    if chunk_bits not in CHUNK_BITS_VALUES:
        raise ValueError(f"unsupported CHUNK_BITS={chunk_bits}")
    if not (is_normal_positive(dividend) and is_normal_positive(divisor)):
        raise DomainExclusion("chunked_reduce requires positive normal inputs")

    reciprocal = reciprocal_up_two(divisor)
    _, _, divisor_exponent = normal_parts(divisor)
    residual = dividend
    stages = 0
    min_progress: int | None = None
    min_quotient_error: int | None = None
    max_quotient_error: int | None = None
    max_corrections = 0
    max_local_quotient = 0

    while residual >= divisor:
        _, _, residual_exponent = normal_parts(residual)
        exponent_gap = residual_exponent - divisor_exponent
        shift = max(exponent_gap - (chunk_bits - 1), 0)
        scaled_residual = scale_normal(residual, -shift)
        scaled_divisor = scale_normal(divisor, shift)

        with np.errstate(all="ignore"):
            scaled_quotient = FP32(scaled_residual * reciprocal)
        if not is_normal_positive(scaled_quotient):
            raise DomainExclusion("local reciprocal product is not positive normal")
        quotient_hat = math.floor(float(scaled_quotient))
        quotient_exact, next_residual = exact_mod_normal(residual, scaled_divisor)
        quotient_error = quotient_hat - quotient_exact

        if quotient_error not in (0, 1):
            raise AssertionError(
                f"local quotient is outside the proven one-sided interval: error={quotient_error} "
                f"a=0x{fp32_bits(dividend):08x} b=0x{fp32_bits(divisor):08x}"
            )
        # The exact local ratio is strictly below 2**CHUNK_BITS.  The one-sided
        # estimate may equal that power of two; that value has only one
        # significant bit and is harmless for the partial-product proof.
        if quotient_hat > 1 << chunk_bits:
            raise AssertionError(f"local quotient {quotient_hat} does not fit CHUNK_BITS={chunk_bits}")

        verify_component_subtractions(residual, scaled_divisor, quotient_hat, quotient_exact, chunk_bits)

        stages += 1
        min_quotient_error = quotient_error if min_quotient_error is None else min(min_quotient_error, quotient_error)
        max_quotient_error = quotient_error if max_quotient_error is None else max(max_quotient_error, quotient_error)
        max_corrections = max(max_corrections, quotient_error)
        max_local_quotient = max(max_local_quotient, quotient_hat)

        if next_residual and next_residual >= divisor:
            _, _, next_exponent = normal_parts(next_residual)
            progress = exponent_gap - (next_exponent - divisor_exponent)
            if progress < chunk_bits - 1:
                raise AssertionError(f"nonterminal progress {progress} is below CHUNK_BITS-1={chunk_bits - 1}")
            min_progress = progress if min_progress is None else min(min_progress, progress)

        residual = next_residual
        if stages > math.ceil(253 / (chunk_bits - 1)):
            raise AssertionError("fixed-stage bound exceeded")

    _, expected = exact_mod_normal(dividend, divisor)
    if fp32_bits(residual) != fp32_bits(expected):
        raise AssertionError(
            f"final mismatch a=0x{fp32_bits(dividend):08x} b=0x{fp32_bits(divisor):08x}: "
            f"got=0x{fp32_bits(residual):08x} expected=0x{fp32_bits(expected):08x}"
        )

    return ReductionResult(
        residual,
        stages,
        min_progress,
        min_quotient_error,
        max_quotient_error,
        max_corrections,
        max_local_quotient,
    )


def boundary_cases(divisors: tuple[np.float32, ...]) -> list[tuple[np.float32, np.float32]]:
    cases = list(deterministic_cases())
    quotients = (1, 2, 15, 16, 17, 255, 256, 257, 4095, 4096, 65535, 65536, 2**20, 2**24)
    for divisor in divisors:
        cases.append((FP32(np.finfo(FP32).max), divisor))
        for quotient in quotients:
            with np.errstate(all="ignore"):
                center = FP32(FP32(quotient) * divisor)
            if not is_normal_positive(center):
                continue
            for dividend in (
                np.nextafter(center, FP32(-math.inf), dtype=FP32),
                center,
                np.nextafter(center, FP32(math.inf), dtype=FP32),
            ):
                if is_normal_positive(dividend):
                    cases.append((dividend, divisor))
    return cases


def evaluate_pair(
    dividend: np.float32,
    divisor: np.float32,
    metrics: dict[int, CandidateMetrics],
) -> None:
    for chunk_bits, candidate in metrics.items():
        try:
            result = chunked_reduce(dividend, divisor, chunk_bits)
            candidate.record(result)
        except DomainExclusion as error:
            candidate.record_exclusion(error)
        except AssertionError:
            candidate.failures += 1
            raise


def run_deterministic(metrics: dict[int, CandidateMetrics], divisors: tuple[np.float32, ...]) -> None:
    cases = boundary_cases(divisors)
    for dividend, divisor in cases:
        evaluate_pair(dividend, divisor, metrics)
    print(f"deterministic_and_boundaries={len(cases)}")


def run_bf16(metrics: dict[int, CandidateMetrics], divisors: tuple[np.float32, ...]) -> None:
    bf16_bits = np.arange(1, 0x7F80, dtype=U32)
    values = (bf16_bits << 16).view(FP32)
    pairs = 0
    for divisor in divisors:
        for dividend in values:
            if is_normal_positive(dividend):
                evaluate_pair(dividend, divisor, metrics)
                pairs += 1
    print(f"bf16_positive_finite_pairs={pairs}")


def run_random(
    metrics: dict[int, CandidateMetrics],
    divisors: tuple[np.float32, ...],
    count: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(count):
        dividend = fp32_from_bits(int(rng.integers(0x00800000, 0x7F7FFFFF, dtype=U32)))
        divisor = divisors[int(rng.integers(0, len(divisors)))]
        evaluate_pair(dividend, divisor, metrics)
    print(f"random_normal_pairs={count} seed=0x{seed:x}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=100_000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x5F91)
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--chunk-bits", type=int, nargs="*", default=list(CHUNK_BITS_VALUES))
    args = parser.parse_args()

    metrics = {chunk_bits: CandidateMetrics(chunk_bits) for chunk_bits in args.chunk_bits}
    run_deterministic(metrics, DEFAULT_DIVISORS)
    if not args.skip_bf16:
        run_bf16(metrics, DEFAULT_DIVISORS)
    run_random(metrics, DEFAULT_DIVISORS, args.random, args.seed)

    for candidate in metrics.values():
        print(candidate.summary())
        if candidate.exclusions:
            print(candidate.exclusion_summary())
    if any(candidate.failures for candidate in metrics.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
