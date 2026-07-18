#!/usr/bin/env python3
"""Exact-reference research harness for scalar SFPI modulo algorithms.

This is a host research tool, not a model of every Tensix special-value detail.
It uses exact rational arithmetic for the mathematical quotient/remainder and
binary32 operations for the reciprocal fast path and exponent-scaled fallback.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

FP32 = np.float32
U32 = np.uint32


def fp32_bits(value: np.float32) -> int:
    return int(np.asarray([value], dtype=FP32).view(U32)[0])


def fp32_from_bits(bits: int) -> np.float32:
    return np.asarray([bits], dtype=U32).view(FP32)[0]


def exact_fraction(value: np.float32) -> Fraction:
    return Fraction(*float(value).as_integer_ratio())


def unbiased_exponent(value: np.float32) -> int:
    bits = fp32_bits(FP32(abs(value)))
    biased = (bits >> 23) & 0xFF
    if biased == 0:
        # Exact exponent for a nonzero binary32 subnormal.
        mantissa = bits & 0x7FFFFF
        return mantissa.bit_length() - 150
    if biased == 0xFF:
        raise ValueError("non-finite value has no finite unbiased exponent")
    return biased - 127


def is_normal_positive(value: np.float32) -> bool:
    bits = fp32_bits(value)
    return 0 < ((bits >> 23) & 0xFF) < 0xFF and (bits >> 31) == 0


def is_power_of_two_normal(value: np.float32) -> bool:
    bits = fp32_bits(value)
    return is_normal_positive(value) and (bits & 0x7FFFFF) == 0


@dataclass(frozen=True)
class FastPathState:
    reciprocal: np.float32
    scaled: np.float32
    q_exact: int
    q_hat: int
    quotient_error: int
    residual_before: np.float32
    residual_after: np.float32
    safe_exp22: bool


def exact_magnitude_remainder(a: np.float32, b: np.float32) -> tuple[int, Fraction]:
    a_exact = exact_fraction(a)
    b_exact = exact_fraction(b)
    q_exact = a_exact // b_exact
    return q_exact, a_exact - q_exact * b_exact


def fast_path(a: np.float32, b: np.float32) -> FastPathState:
    q_exact, _ = exact_magnitude_remainder(a, b)
    with np.errstate(all="ignore"):
        reciprocal = FP32(FP32(1.0) / b)
        scaled = FP32(a * reciprocal)
    if not np.isfinite(scaled):
        return FastPathState(reciprocal, scaled, q_exact, 0, -q_exact, FP32(np.nan), FP32(np.nan), False)

    q_hat = math.floor(float(scaled))
    # Host approximation of the SFPMAD residual. Quotient-error classification
    # does not depend on the exact residual rounding model.
    with np.errstate(all="ignore"):
        residual = FP32(a - FP32(FP32(q_hat) * b))
        corrected = FP32(residual - b) if residual >= b else residual
        corrected = FP32(corrected + b) if corrected < FP32(0.0) else corrected

    safe_exp22 = scaled >= FP32(0.0) and unbiased_exponent(scaled) < 22 if scaled != FP32(0.0) else True
    return FastPathState(
        reciprocal,
        scaled,
        q_exact,
        q_hat,
        q_hat - q_exact,
        residual,
        corrected,
        safe_exp22,
    )


def binary_reduce(a: np.float32, b: np.float32) -> np.float32:
    """Radix-2 exponent-scaled reduction for positive finite normal a and b.

    Each subtraction is between values whose ratio lies in [1, 2), making it
    exact under IEEE RNE by Sterbenz's lemma, except where FTZ/subnormal behavior
    becomes relevant.
    """

    if a < b:
        return a
    if a == b:
        return FP32(0.0)
    if not (is_normal_positive(a) and is_normal_positive(b)):
        raise ValueError("binary_reduce currently models positive normal operands only")

    exponent_gap = unbiased_exponent(a) - unbiased_exponent(b)
    divisor_bits = fp32_bits(b)
    divisor_biased_exp = (divisor_bits >> 23) & 0xFF
    scaled_exp = divisor_biased_exp + exponent_gap
    if scaled_exp >= 0xFF:
        exponent_gap -= 1
        scaled_exp -= 1
    scaled_bits = (divisor_bits & 0x807FFFFF) | (scaled_exp << 23)
    divisor = fp32_from_bits(scaled_bits)
    if divisor > a:
        divisor = FP32(divisor * FP32(0.5))
        exponent_gap -= 1

    residual = a
    while exponent_gap >= 0:
        if residual >= divisor:
            residual = FP32(residual - divisor)
        divisor = FP32(divisor * FP32(0.5))
        exponent_gap -= 1
    return residual


def fraction_to_fp32_exact(value: Fraction) -> np.float32:
    # Conversion through float64 is exact for a difference of binary32 values
    # in the tested normal-domain cases. Equality is checked again as a Fraction.
    result = FP32(float(value))
    if exact_fraction(result) != value:
        raise ValueError(f"exact remainder is not represented by normal FP32: {value}")
    return result


def check_pair(a: np.float32, b: np.float32) -> tuple[FastPathState, bool]:
    state = fast_path(a, b)
    _, exact_remainder = exact_magnitude_remainder(a, b)
    reduced = binary_reduce(a, b) if a >= b else a
    try:
        expected = fraction_to_fp32_exact(exact_remainder)
        fallback_ok = fp32_bits(reduced) == fp32_bits(expected)
    except ValueError:
        fallback_ok = exact_fraction(reduced) == exact_remainder
    if state.safe_exp22 and abs(state.quotient_error) > 1:
        raise AssertionError(
            f"unsafe classifier: a=0x{fp32_bits(a):08x}, b=0x{fp32_bits(b):08x}, " f"q_error={state.quotient_error}"
        )
    return state, fallback_ok


def deterministic_cases() -> list[tuple[np.float32, np.float32]]:
    decimal = [3.0, 5.0, 7.0, 10.0, 0.1, 0.3, 0.003]
    dividends = [50331656.0, 83886104.0, 116040560.0, 167772208.0, 50855936.0]
    cases = [(FP32(a), FP32(b)) for a, b in zip(dividends, decimal[:4] + [3.0])]
    for b_value in decimal:
        b = FP32(b_value)
        for quotient in (2**20 - 1, 2**22 - 1, 2**23, 2**24, 2**25):
            center = FP32(FP32(quotient) * b)
            cases.extend(
                (candidate, b)
                for candidate in (
                    np.nextafter(center, FP32(-math.inf), dtype=FP32),
                    center,
                    np.nextafter(center, FP32(math.inf), dtype=FP32),
                )
                if is_normal_positive(candidate)
            )
    return cases


def run_deterministic() -> None:
    failures = 0
    unsafe = 0
    for a, b in deterministic_cases():
        state, fallback_ok = check_pair(a, b)
        unsafe += int(not state.safe_exp22)
        if not fallback_ok:
            failures += 1
            print(f"fallback failure a=0x{fp32_bits(a):08x} b=0x{fp32_bits(b):08x}")
    print(f"deterministic cases={len(deterministic_cases())} unsafe={unsafe} fallback_failures={failures}")
    if failures:
        raise SystemExit(1)


def run_bf16_exhaustive(divisors: list[np.float32]) -> None:
    bf16_bits = np.arange(1, 0x7F80, dtype=U32)
    values = (bf16_bits << 16).view(FP32)
    safe_checked = 0
    fallback_checked = 0
    for b in divisors:
        for a in values:
            if not is_normal_positive(a):
                continue
            state = fast_path(a, b)
            if state.safe_exp22:
                safe_checked += 1
                if abs(state.quotient_error) > 1:
                    raise AssertionError(f"BF16 classifier failure a=0x{fp32_bits(a):08x} b=0x{fp32_bits(b):08x}")
            elif np.isfinite(state.scaled):
                _, fallback_ok = check_pair(a, b)
                fallback_checked += 1
                if not fallback_ok:
                    raise AssertionError(f"BF16 fallback failure a=0x{fp32_bits(a):08x} b=0x{fp32_bits(b):08x}")
    print(f"BF16 exhaustive safe={safe_checked} fallback={fallback_checked}")


def run_random(count: int, seed: int, divisors: list[np.float32]) -> None:
    rng = np.random.default_rng(seed)
    safe_checked = 0
    fallback_checked = 0
    for _ in range(count):
        a = fp32_from_bits(int(rng.integers(0x00800000, 0x7F7FFFFF, dtype=U32)))
        b = divisors[int(rng.integers(0, len(divisors)))]
        state = fast_path(a, b)
        if state.safe_exp22:
            safe_checked += 1
            if abs(state.quotient_error) > 1:
                raise AssertionError(f"random classifier failure a=0x{fp32_bits(a):08x} b=0x{fp32_bits(b):08x}")
        elif np.isfinite(state.scaled):
            _, fallback_ok = check_pair(a, b)
            fallback_checked += 1
            if not fallback_ok:
                raise AssertionError(f"random fallback failure a=0x{fp32_bits(a):08x} b=0x{fp32_bits(b):08x}")
    print(f"random safe={safe_checked} fallback={fallback_checked} seed=0x{seed:x}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=100_000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x5F91)
    parser.add_argument("--skip-bf16", action="store_true")
    args = parser.parse_args()

    divisors = [FP32(value) for value in (2.0, 3.0, 5.0, 7.0, 10.0, 0.1, 0.3, 0.003)]
    run_deterministic()
    if not args.skip_bf16:
        run_bf16_exhaustive(divisors)
    run_random(args.random, args.seed, divisors)


if __name__ == "__main__":
    main()
