// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Multithreaded positive-FP32 verifier for the test-only scalar-specialized
// modulo fixed schedule.  This is a host research tool, not production code.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kPositiveFiniteEnd = 0x7f800000u;
constexpr std::uint32_t kPositiveNormalBegin = 0x00800000u;
constexpr std::uint32_t kMantissaMask = 0x007fffffu;
constexpr int kInitialDivisorExponent = 112;
constexpr int kChunkStep = 15;
constexpr int kDefaultWorkingExponent = 111;

enum class Outcome : std::uint32_t {
    Passed,
    InputSubnormal,
    UnsupportedSpecialValue,
    ScaledDivisorNotNormal,
    InverseNotNormal,
    ScaledQuotientNotNormal,
    QuotientOutOfRange,
    QuotientError,
    SplitComponentSubnormal,
    PartialProductPrecision,
    PartialProductOverflow,
    IntermediatePrecision,
    IntermediateSubnormal,
    IntermediateOverflow,
    ExactRemainderSubnormal,
    PostReductionOverflow,
    CorrectionMismatch,
    FinalMismatch,
    InternalInvariant,
    Count,
};

constexpr std::array<std::string_view, static_cast<std::size_t>(Outcome::Count)> kOutcomeNames = {
    "Passed",
    "InputSubnormal",
    "UnsupportedSpecialValue",
    "ScaledDivisorNotNormal",
    "InverseNotNormal",
    "ScaledQuotientNotNormal",
    "QuotientOutOfRange",
    "QuotientError",
    "SplitComponentSubnormal",
    "PartialProductPrecision",
    "PartialProductOverflow",
    "IntermediatePrecision",
    "IntermediateSubnormal",
    "IntermediateOverflow",
    "ExactRemainderSubnormal",
    "PostReductionOverflow",
    "CorrectionMismatch",
    "FinalMismatch",
    "InternalInvariant",
};

enum class SubnormalPoint : std::uint32_t {
    None,
    InitialScaling,
    QuotientProduct,
    DivisorHigh,
    DivisorLow,
    HighProduct,
    LowProduct,
    AfterHighSubtract,
    AfterLowSubtract,
    AfterCorrection,
    InterstageScale,
    FinalResult,
    Count,
};

constexpr std::array<std::string_view, static_cast<std::size_t>(SubnormalPoint::Count)> kSubnormalPointNames = {
    "None",
    "InitialScaling",
    "QuotientProduct",
    "DivisorHigh",
    "DivisorLow",
    "HighProduct",
    "LowProduct",
    "AfterHighSubtract",
    "AfterLowSubtract",
    "AfterCorrection",
    "InterstageScale",
    "FinalResult",
};

enum class ValueClass : std::uint32_t {
    Zero,
    Subnormal,
    Normal,
    Special,
    Count,
};

constexpr std::array<std::string_view, static_cast<std::size_t>(ValueClass::Count)> kValueClassNames = {
    "Zero",
    "Subnormal",
    "Normal",
    "Special",
};

struct Options {
    std::uint32_t divisor_bits = 0x40400000u;
    std::uint64_t input_begin = kPositiveNormalBegin;
    std::uint64_t input_end = kPositiveFiniteEnd;
    std::uint64_t shard = 0;
    std::uint64_t num_shards = 1;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::size_t max_records = 64;
    std::string dump_failures;
    std::string dump_exclusions;
    bool stop_on_failure = false;
    bool prehalve_max_exponent = false;
    bool exponent_stationary = false;
    int working_exponent = kDefaultWorkingExponent;
};

struct FloatParts {
    std::uint32_t significand = 0;
    int unit_exponent = 0;
    int unbiased_exponent = 0;
    bool normal = false;
    bool subnormal = false;
    bool zero = false;
};

struct Config {
    std::uint32_t divisor_bits = 0;
    std::uint32_t reduction_divisor_bits = 0;
    std::uint32_t reciprocal_up_bits = 0;
    std::uint32_t initial_inverse_bits = 0;
    unsigned high_mantissa = 0;
    int start_shift = 0;
    int divisor_exponent = 0;
    int working_exponent = 0;
    int final_exponent_shift = 0;
    bool exponent_stationary = false;
};

struct EvaluationMetadata {
    int stage_index = -1;
    int exact_residual_exponent = std::numeric_limits<int>::min();
    SubnormalPoint subnormal_point = SubnormalPoint::None;
    bool negative_subnormal = false;
};

struct Record {
    Outcome outcome = Outcome::Passed;
    std::uint32_t input_bits = 0;
    std::uint32_t divisor_bits = 0;
    std::uint32_t got_bits = 0;
    std::uint32_t expected_bits = 0;
    int stage_shift = -1;
    std::uint32_t quotient_hat = 0;
    std::uint32_t quotient_exact = 0;
    std::uint64_t detail = 0;
    int stage_index = -1;
    int divisor_exponent = 0;
    int exact_residual_exponent = std::numeric_limits<int>::min();
    SubnormalPoint subnormal_point = SubnormalPoint::None;
    ValueClass final_exact_class = ValueClass::Zero;
    bool negative_subnormal = false;
};

struct Evaluation {
    Outcome outcome = Outcome::Passed;
    Record record;
};

struct ThreadResult {
    std::array<std::uint64_t, static_cast<std::size_t>(Outcome::Count)> counts{};
    std::array<std::uint64_t, static_cast<std::size_t>(SubnormalPoint::Count)> subnormal_point_counts{};
    std::array<
        std::array<std::uint64_t, static_cast<std::size_t>(ValueClass::Count)>,
        static_cast<std::size_t>(SubnormalPoint::Count)>
        subnormal_final_class_counts{};
    std::uint64_t negative_subnormal_count = 0;
    std::vector<Record> failures;
    std::vector<Record> exclusions;
};

std::uint64_t parse_u64(const std::string& value) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 0);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid integer: " + value);
    }
    return parsed;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --divisor-bits N          positive scalar FP32 bits (default 0x40400000)\n"
              << "  --input-begin-bits N      inclusive positive input bits\n"
              << "  --input-end-bits N        exclusive positive input bits\n"
              << "  --exhaustive-inputs       use all positive normal inputs\n"
              << "  --include-subnormals      start at +0 instead of smallest normal\n"
              << "  --include-specials        extend through positive Inf/NaN encodings\n"
              << "  --shard N                 zero-based shard index\n"
              << "  --num-shards N            number of equal bit-range shards\n"
              << "  --threads N               worker threads\n"
              << "  --dump-failures PATH      write proof/final failures as TSV\n"
              << "  --dump-exclusions PATH    write domain exclusions as TSV\n"
              << "  --max-records N           maximum records per output (default 64)\n"
              << "  --prehalve-max-exponent   test exact a/2 reduction for exponent-127 inputs\n"
              << "  --exponent-stationary     keep the divisor at a constant normalized exponent\n"
              << "  --working-exponent N      normalized divisor exponent (default 111)\n"
              << "  --stop-on-failure         stop after a proof/final failure\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto require_value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument("missing value after " + argument);
            }
            return argv[index];
        };

        if (argument == "--divisor-bits") {
            options.divisor_bits = static_cast<std::uint32_t>(parse_u64(require_value()));
        } else if (argument == "--input-begin-bits") {
            options.input_begin = parse_u64(require_value());
        } else if (argument == "--input-end-bits") {
            options.input_end = parse_u64(require_value());
        } else if (argument == "--exhaustive-inputs") {
            options.input_begin = kPositiveNormalBegin;
            options.input_end = kPositiveFiniteEnd;
        } else if (argument == "--include-subnormals") {
            options.input_begin = 0;
        } else if (argument == "--include-specials") {
            options.input_end = 0x80000000ull;
        } else if (argument == "--shard") {
            options.shard = parse_u64(require_value());
        } else if (argument == "--num-shards") {
            options.num_shards = parse_u64(require_value());
        } else if (argument == "--threads") {
            options.threads = static_cast<unsigned>(parse_u64(require_value()));
        } else if (argument == "--dump-failures") {
            options.dump_failures = require_value();
        } else if (argument == "--dump-exclusions") {
            options.dump_exclusions = require_value();
        } else if (argument == "--max-records") {
            options.max_records = static_cast<std::size_t>(parse_u64(require_value()));
        } else if (argument == "--stop-on-failure") {
            options.stop_on_failure = true;
        } else if (argument == "--prehalve-max-exponent") {
            options.prehalve_max_exponent = true;
        } else if (argument == "--exponent-stationary") {
            options.exponent_stationary = true;
        } else if (argument == "--working-exponent") {
            options.working_exponent = static_cast<int>(std::stoll(require_value()));
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    if (options.num_shards == 0 || options.shard >= options.num_shards) {
        throw std::invalid_argument("require 0 <= shard < num-shards");
    }
    if (options.threads == 0) {
        throw std::invalid_argument("threads must be positive");
    }
    if (options.input_begin > options.input_end || options.input_end > 0x80000000ull) {
        throw std::invalid_argument("invalid positive input range");
    }
    if (options.working_exponent < -126 || options.working_exponent > 111) {
        throw std::invalid_argument("working exponent must be in [-126, 111]");
    }
    if (options.exponent_stationary && options.working_exponent != kDefaultWorkingExponent) {
        throw std::invalid_argument("the current exponent-stationary proof requires working exponent 111");
    }
    return options;
}

FloatParts decode_float(std::uint32_t bits) {
    FloatParts parts;
    const std::uint32_t biased = (bits >> 23) & 0xffu;
    const std::uint32_t mantissa = bits & kMantissaMask;
    if (biased == 0) {
        parts.zero = mantissa == 0;
        parts.subnormal = mantissa != 0;
        parts.significand = mantissa;
        parts.unit_exponent = -149;
        parts.unbiased_exponent = -126;
    } else if (biased != 0xffu) {
        parts.normal = true;
        parts.significand = 0x00800000u | mantissa;
        parts.unbiased_exponent = static_cast<int>(biased) - 127;
        parts.unit_exponent = parts.unbiased_exponent - 23;
    }
    return parts;
}

bool is_positive_normal(std::uint32_t bits) {
    return bits < 0x80000000u && ((bits >> 23) & 0xffu) > 0 && ((bits >> 23) & 0xffu) < 0xffu;
}

std::optional<std::uint32_t> scale_normal_bits(std::uint32_t bits, int delta) {
    const int biased = static_cast<int>((bits >> 23) & 0xffu);
    const int scaled = biased + delta;
    if (bits >= 0x80000000u || biased <= 0 || biased >= 0xff || scaled <= 0 || scaled >= 0xff) {
        return std::nullopt;
    }
    return (bits & 0x807fffffu) | (static_cast<std::uint32_t>(scaled) << 23);
}

int floor_log2(std::uint64_t value) { return 63 - std::countl_zero(value); }

int precision_span(std::uint64_t value) {
    if (value == 0) {
        return 0;
    }
    return floor_log2(value) + 1 - std::countr_zero(value);
}

std::uint64_t magnitude(std::int64_t value) {
    return value < 0 ? static_cast<std::uint64_t>(-value) : static_cast<std::uint64_t>(value);
}

std::uint64_t pow2_mod(unsigned exponent, std::uint32_t modulus) {
    std::uint64_t result = 1 % modulus;
    std::uint64_t base = 2 % modulus;
    while (exponent != 0) {
        if (exponent & 1u) {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exponent >>= 1;
    }
    return result;
}

std::optional<std::uint32_t> compose_exact_bits(std::uint64_t significand, int unit_exponent) {
    if (significand == 0) {
        return 0u;
    }
    const int top_exponent = unit_exponent + floor_log2(significand);
    if (top_exponent > 127) {
        return std::nullopt;
    }
    if (top_exponent >= -126) {
        const int shift = 23 - floor_log2(significand);
        if (shift < 0) {
            return std::nullopt;
        }
        const std::uint64_t normalized = significand << shift;
        return (static_cast<std::uint32_t>(top_exponent + 127) << 23) |
               (static_cast<std::uint32_t>(normalized) & kMantissaMask);
    }

    const int shift = unit_exponent + 149;
    if (shift < 0 || shift >= 64) {
        return std::nullopt;
    }
    const std::uint64_t subnormal = significand << shift;
    if (subnormal >= 0x00800000ull) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(subnormal);
}

std::uint64_t round_shift_right_even(std::uint64_t value, unsigned shift) {
    if (shift == 0) {
        return value;
    }
    if (shift > 64) {
        return 0;
    }
    if (shift == 64) {
        return value > (std::uint64_t{1} << 63) ? 1 : 0;
    }

    const std::uint64_t truncated = value >> shift;
    const std::uint64_t remainder = value & ((std::uint64_t{1} << shift) - 1);
    const std::uint64_t halfway = std::uint64_t{1} << (shift - 1);
    return truncated +
           static_cast<std::uint64_t>(remainder > halfway || (remainder == halfway && (truncated & 1u) != 0));
}

std::optional<std::uint32_t> compose_rne_bits(std::uint64_t significand, int unit_exponent, bool sign = false) {
    const std::uint32_t sign_bits = sign ? 0x80000000u : 0u;
    if (significand == 0) {
        return sign_bits;
    }

    int top_exponent = unit_exponent + floor_log2(significand);
    if (top_exponent > 127) {
        return std::nullopt;
    }
    if (top_exponent >= -126) {
        const int precision_shift = floor_log2(significand) - 23;
        std::uint64_t normalized = precision_shift > 0
                                       ? round_shift_right_even(significand, static_cast<unsigned>(precision_shift))
                                       : significand << -precision_shift;
        if (normalized == 0x01000000ull) {
            normalized >>= 1;
            ++top_exponent;
            if (top_exponent > 127) {
                return std::nullopt;
            }
        }
        return sign_bits | (static_cast<std::uint32_t>(top_exponent + 127) << 23) |
               (static_cast<std::uint32_t>(normalized) & kMantissaMask);
    }

    const int subnormal_shift = unit_exponent + 149;
    const std::uint64_t subnormal = subnormal_shift >= 0
                                        ? significand << subnormal_shift
                                        : round_shift_right_even(significand, static_cast<unsigned>(-subnormal_shift));
    if (subnormal >= 0x00800000ull) {
        return sign_bits | kPositiveNormalBegin;
    }
    return sign_bits | static_cast<std::uint32_t>(subnormal);
}

std::optional<std::uint32_t> pack_scaled_fp32_exact(
    std::uint32_t normalized_value_bits, int physical_exponent_shift, bool sign = false) {
    if (normalized_value_bits == 0) {
        return sign ? 0x80000000u : 0u;
    }
    const FloatParts normalized = decode_float(normalized_value_bits);
    if (!normalized.normal) {
        return std::nullopt;
    }
    return compose_rne_bits(normalized.significand, normalized.unit_exponent + physical_exponent_shift, sign);
}

ValueClass classify_bits(std::uint32_t bits) {
    const std::uint32_t magnitude_bits = bits & 0x7fffffffu;
    const std::uint32_t biased = (magnitude_bits >> 23) & 0xffu;
    if (biased == 0) {
        return magnitude_bits == 0 ? ValueClass::Zero : ValueClass::Subnormal;
    }
    return biased == 0xffu ? ValueClass::Special : ValueClass::Normal;
}

void validate_exact_packer() {
    struct PackCase {
        std::uint32_t normalized_bits;
        int exponent_shift;
        bool sign;
        std::optional<std::uint32_t> expected_bits;
    };
    constexpr std::array<PackCase, 8> cases = {{
        {0x00000000u, 0, false, 0x00000000u},
        {0x00000000u, 0, true, 0x80000000u},
        {0x3f800000u, 0, false, 0x3f800000u},
        {0x3f800000u, -149, false, 0x00000001u},
        // Halfway between the largest subnormal and the smallest normal;
        // ties-to-even must carry into the smallest normal.
        {0x3fffffffu, -127, false, 0x00800000u},
        // A halfway case whose truncated subnormal is already even.
        {0x3ffffffdu, -127, false, 0x007ffffeu},
        {0x3fffffffu, -128, false, 0x00400000u},
        {0x3f800000u, 128, false, std::nullopt},
    }};

    for (const PackCase& pack_case : cases) {
        const auto actual = pack_scaled_fp32_exact(pack_case.normalized_bits, pack_case.exponent_shift, pack_case.sign);
        if (actual != pack_case.expected_bits) {
            throw std::logic_error("exact FP32 packer self-test failed");
        }
    }
}

std::optional<std::uint32_t> exact_mod_bits(std::uint32_t input_bits, std::uint32_t divisor_bits) {
    if (input_bits == 0) {
        return 0u;
    }
    if (input_bits < divisor_bits) {
        return input_bits;
    }

    const FloatParts input = decode_float(input_bits);
    const FloatParts divisor = decode_float(divisor_bits);
    if (!(input.normal || input.subnormal) || !(divisor.normal || divisor.subnormal)) {
        return std::nullopt;
    }
    const int delta = input.unit_exponent - divisor.unit_exponent;
    if (delta < 0) {
        return std::nullopt;
    }
    const std::uint64_t factor = pow2_mod(static_cast<unsigned>(delta), divisor.significand);
    const std::uint64_t remainder = (static_cast<std::uint64_t>(input.significand) * factor) % divisor.significand;
    return compose_exact_bits(remainder, divisor.unit_exponent);
}

std::uint32_t nearest_away_uint16(std::uint32_t positive_float_bits) {
    const int exponent = static_cast<int>((positive_float_bits >> 23) & 0xffu) - 127;
    if (exponent < -1) {
        return 0;
    }
    if (exponent >= 16) {
        return 65535;
    }

    std::uint64_t magnitude_bits = 0x00800000u | (positive_float_bits & kMantissaMask);
    if (exponent >= 0) {
        magnitude_bits <<= exponent;
    } else {
        magnitude_bits >>= -exponent;
    }
    std::uint64_t rounded = (magnitude_bits >> 23) + ((magnitude_bits & 0x007fffffu) >= 0x00400000u);
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(rounded, 65535));
}

std::optional<std::uint32_t> make_biased_reciprocal(std::uint32_t divisor_bits) {
    const float divisor = std::bit_cast<float>(divisor_bits);
    float reciprocal = 1.0f / divisor;
    if ((divisor_bits & kMantissaMask) != 0) {
        reciprocal = std::nextafter(reciprocal, std::numeric_limits<float>::infinity());
        reciprocal = std::nextafter(reciprocal, std::numeric_limits<float>::infinity());
    }
    const std::uint32_t reciprocal_bits = std::bit_cast<std::uint32_t>(reciprocal);
    if (!is_positive_normal(reciprocal_bits)) {
        return std::nullopt;
    }
    return reciprocal_bits;
}

std::optional<Config> make_config(std::uint32_t divisor_bits, bool exponent_stationary, int working_exponent) {
    if (!is_positive_normal(divisor_bits)) {
        return std::nullopt;
    }
    const FloatParts divisor_parts = decode_float(divisor_bits);

    Config config;
    config.divisor_bits = divisor_bits;
    config.divisor_exponent = divisor_parts.unbiased_exponent;
    config.working_exponent = working_exponent;
    config.exponent_stationary = exponent_stationary;
    config.high_mantissa = ((divisor_bits & kMantissaMask) & ~0xfffu) >> 11;

    if (exponent_stationary) {
        config.start_shift = std::max(working_exponent - divisor_parts.unbiased_exponent, 0);
        config.final_exponent_shift = divisor_parts.unbiased_exponent - working_exponent;
        const auto normalized_divisor =
            scale_normal_bits(divisor_bits, working_exponent - divisor_parts.unbiased_exponent);
        if (!normalized_divisor) {
            return std::nullopt;
        }
        const auto reciprocal = make_biased_reciprocal(*normalized_divisor);
        if (!reciprocal) {
            return std::nullopt;
        }
        config.reduction_divisor_bits = *normalized_divisor;
        config.reciprocal_up_bits = *reciprocal;
        config.initial_inverse_bits = *reciprocal;
        return config;
    }

    config.start_shift = std::max(kInitialDivisorExponent - divisor_parts.unbiased_exponent, 0);
    config.final_exponent_shift = 0;
    const auto reciprocal = make_biased_reciprocal(divisor_bits);
    if (!reciprocal) {
        return std::nullopt;
    }
    const auto initial_inverse = scale_normal_bits(*reciprocal, -config.start_shift);
    const auto reduction_divisor = scale_normal_bits(divisor_bits, config.start_shift);
    if (!initial_inverse || !reduction_divisor) {
        return std::nullopt;
    }
    config.reduction_divisor_bits = *reduction_divisor;
    config.reciprocal_up_bits = *reciprocal;
    config.initial_inverse_bits = *initial_inverse;
    return config;
}

bool is_failure(Outcome outcome) {
    switch (outcome) {
        case Outcome::QuotientOutOfRange:
        case Outcome::QuotientError:
        case Outcome::PartialProductPrecision:
        case Outcome::IntermediatePrecision:
        case Outcome::CorrectionMismatch:
        case Outcome::FinalMismatch:
        case Outcome::InternalInvariant: return true;
        default: return false;
    }
}

Evaluation make_evaluation(
    Outcome outcome,
    const Config& config,
    std::uint32_t input_bits,
    std::uint32_t got_bits,
    std::uint32_t expected_bits,
    int shift = -1,
    std::uint32_t quotient_hat = 0,
    std::uint32_t quotient_exact = 0,
    std::uint64_t detail = 0,
    EvaluationMetadata metadata = {}) {
    Evaluation evaluation;
    evaluation.outcome = outcome;
    evaluation.record.outcome = outcome;
    evaluation.record.input_bits = input_bits;
    evaluation.record.divisor_bits = config.divisor_bits;
    evaluation.record.got_bits = got_bits;
    evaluation.record.expected_bits = expected_bits;
    evaluation.record.stage_shift = shift;
    evaluation.record.quotient_hat = quotient_hat;
    evaluation.record.quotient_exact = quotient_exact;
    evaluation.record.detail = detail;
    evaluation.record.stage_index = metadata.stage_index;
    evaluation.record.divisor_exponent = config.divisor_exponent;
    evaluation.record.exact_residual_exponent = metadata.exact_residual_exponent;
    evaluation.record.subnormal_point = metadata.subnormal_point;
    evaluation.record.final_exact_class = classify_bits(expected_bits);
    evaluation.record.negative_subnormal = metadata.negative_subnormal;
    return evaluation;
}

Evaluation evaluate(std::uint32_t input_bits, const Config& config, bool prehalve_max_exponent) {
    if (input_bits >= kPositiveFiniteEnd) {
        return make_evaluation(Outcome::UnsupportedSpecialValue, config, input_bits, 0, 0);
    }
    const auto expected_optional = exact_mod_bits(input_bits, config.divisor_bits);
    if (!expected_optional) {
        return make_evaluation(Outcome::InternalInvariant, config, input_bits, 0, 0);
    }
    const std::uint32_t expected_bits = *expected_optional;
    if (input_bits != 0 && input_bits < kPositiveNormalBegin) {
        return make_evaluation(Outcome::InputSubnormal, config, input_bits, input_bits, expected_bits);
    }

    if (config.exponent_stationary && input_bits < config.divisor_bits) {
        return make_evaluation(Outcome::Passed, config, input_bits, input_bits, expected_bits);
    }

    std::optional<std::uint32_t> divisor_bits = config.reduction_divisor_bits;
    std::uint32_t inverse_bits = config.initial_inverse_bits;
    std::uint32_t residual_bits = input_bits;
    bool prehalved = false;
    const bool use_prehalve = config.exponent_stationary || prehalve_max_exponent;
    if (use_prehalve && input_bits != 0 && decode_float(input_bits).unbiased_exponent == 127) {
        const auto half_input = scale_normal_bits(input_bits, -1);
        if (!half_input) {
            return make_evaluation(Outcome::InternalInvariant, config, input_bits, 0, expected_bits);
        }
        residual_bits = *half_input;
        prehalved = true;
    }

    if (config.exponent_stationary && config.divisor_exponent > config.working_exponent && residual_bits != 0) {
        const auto normalized_input =
            scale_normal_bits(residual_bits, config.working_exponent - config.divisor_exponent);
        if (!normalized_input) {
            return make_evaluation(
                Outcome::IntermediateSubnormal,
                config,
                input_bits,
                residual_bits,
                expected_bits,
                config.start_shift,
                0,
                0,
                residual_bits,
                {.subnormal_point = SubnormalPoint::InitialScaling});
        }
        residual_bits = *normalized_input;
    }
    int shift = config.start_shift;
    int stage_index = 0;

    while (true) {
        if (!is_positive_normal(*divisor_bits)) {
            return make_evaluation(
                Outcome::ScaledDivisorNotNormal, config, input_bits, residual_bits, expected_bits, shift);
        }
        if (!is_positive_normal(inverse_bits)) {
            return make_evaluation(Outcome::InverseNotNormal, config, input_bits, residual_bits, expected_bits, shift);
        }

        if (residual_bits >= *divisor_bits) {
            const float residual = std::bit_cast<float>(residual_bits);
            const float inverse = std::bit_cast<float>(inverse_bits);
            const float scaled_quotient = residual * inverse;
            const std::uint32_t scaled_quotient_bits = std::bit_cast<std::uint32_t>(scaled_quotient);
            if (!is_positive_normal(scaled_quotient_bits)) {
                const ValueClass quotient_class = classify_bits(scaled_quotient_bits);
                return make_evaluation(
                    Outcome::ScaledQuotientNotNormal,
                    config,
                    input_bits,
                    residual_bits,
                    expected_bits,
                    shift,
                    0,
                    0,
                    scaled_quotient_bits,
                    {
                        .stage_index = stage_index,
                        .subnormal_point = quotient_class == ValueClass::Subnormal ? SubnormalPoint::QuotientProduct
                                                                                   : SubnormalPoint::None,
                    });
            }

            const std::uint32_t quotient_hat = nearest_away_uint16(scaled_quotient_bits);
            const FloatParts residual_parts = decode_float(residual_bits);
            const FloatParts divisor_parts = decode_float(*divisor_bits);
            const int unit_delta = residual_parts.unit_exponent - divisor_parts.unit_exponent;
            if (unit_delta < 0 || unit_delta > 31) {
                return make_evaluation(
                    Outcome::InternalInvariant,
                    config,
                    input_bits,
                    residual_bits,
                    expected_bits,
                    shift,
                    quotient_hat,
                    0,
                    static_cast<std::uint64_t>(unit_delta));
            }
            const std::uint64_t residual_integer = static_cast<std::uint64_t>(residual_parts.significand) << unit_delta;
            const std::uint32_t quotient_exact =
                static_cast<std::uint32_t>(residual_integer / divisor_parts.significand);
            const std::uint64_t exact_remainder = residual_integer % divisor_parts.significand;
            if (quotient_hat > 65535) {
                return make_evaluation(
                    Outcome::QuotientOutOfRange,
                    config,
                    input_bits,
                    residual_bits,
                    expected_bits,
                    shift,
                    quotient_hat,
                    quotient_exact);
            }
            const std::int64_t quotient_error =
                static_cast<std::int64_t>(quotient_hat) - static_cast<std::int64_t>(quotient_exact);
            if (quotient_error != 0 && quotient_error != 1) {
                return make_evaluation(
                    Outcome::QuotientError,
                    config,
                    input_bits,
                    residual_bits,
                    expected_bits,
                    shift,
                    quotient_hat,
                    quotient_exact,
                    scaled_quotient_bits);
            }

            std::int64_t transient = static_cast<std::int64_t>(residual_integer);
            const std::array<std::uint32_t, 2> components = {
                divisor_parts.significand & ~0xfffu,
                divisor_parts.significand & 0xfffu,
            };
            for (std::size_t component_index = 0; component_index < components.size(); ++component_index) {
                const std::uint32_t component = components[component_index];
                if (component == 0) {
                    continue;
                }
                const int component_exponent = divisor_parts.unit_exponent + floor_log2(component);
                if (component_exponent < -126) {
                    return make_evaluation(
                        Outcome::SplitComponentSubnormal,
                        config,
                        input_bits,
                        residual_bits,
                        expected_bits,
                        shift,
                        quotient_hat,
                        quotient_exact,
                        component,
                        {
                            .stage_index = stage_index,
                            .exact_residual_exponent = component_exponent,
                            .subnormal_point =
                                component_index == 0 ? SubnormalPoint::DivisorHigh : SubnormalPoint::DivisorLow,
                        });
                }
                const std::uint64_t product = static_cast<std::uint64_t>(quotient_hat) * component;
                if (precision_span(product) > 28) {
                    return make_evaluation(
                        Outcome::PartialProductPrecision,
                        config,
                        input_bits,
                        residual_bits,
                        expected_bits,
                        shift,
                        quotient_hat,
                        quotient_exact,
                        product);
                }
                if (product != 0) {
                    const int product_exponent = divisor_parts.unit_exponent + floor_log2(product);
                    if (product_exponent < -126) {
                        return make_evaluation(
                            Outcome::IntermediateSubnormal,
                            config,
                            input_bits,
                            residual_bits,
                            expected_bits,
                            shift,
                            quotient_hat,
                            quotient_exact,
                            product,
                            {
                                .stage_index = stage_index,
                                .exact_residual_exponent = product_exponent,
                                .subnormal_point =
                                    component_index == 0 ? SubnormalPoint::HighProduct : SubnormalPoint::LowProduct,
                            });
                    }
                    if (product_exponent >= 128) {
                        return make_evaluation(
                            Outcome::PartialProductOverflow,
                            config,
                            input_bits,
                            residual_bits,
                            expected_bits,
                            shift,
                            quotient_hat,
                            quotient_exact,
                            product,
                            {
                                .stage_index = stage_index,
                                .exact_residual_exponent = product_exponent,
                            });
                    }
                }

                transient -= static_cast<std::int64_t>(product);
                const std::uint64_t transient_magnitude = magnitude(transient);
                if (precision_span(transient_magnitude) > 24) {
                    return make_evaluation(
                        Outcome::IntermediatePrecision,
                        config,
                        input_bits,
                        residual_bits,
                        expected_bits,
                        shift,
                        quotient_hat,
                        quotient_exact,
                        transient_magnitude);
                }
                if (transient_magnitude != 0) {
                    const int transient_exponent = divisor_parts.unit_exponent + floor_log2(transient_magnitude);
                    if (transient_exponent < -126) {
                        return make_evaluation(
                            Outcome::IntermediateSubnormal,
                            config,
                            input_bits,
                            residual_bits,
                            expected_bits,
                            shift,
                            quotient_hat,
                            quotient_exact,
                            transient_magnitude,
                            {
                                .stage_index = stage_index,
                                .exact_residual_exponent = transient_exponent,
                                .subnormal_point = component_index == 0 ? SubnormalPoint::AfterHighSubtract
                                                                        : SubnormalPoint::AfterLowSubtract,
                                .negative_subnormal = transient < 0,
                            });
                    }
                    if (transient_exponent > 127) {
                        return make_evaluation(
                            Outcome::IntermediateOverflow,
                            config,
                            input_bits,
                            residual_bits,
                            expected_bits,
                            shift,
                            quotient_hat,
                            quotient_exact,
                            transient_magnitude);
                    }
                }
            }

            if (quotient_error == 1) {
                transient += divisor_parts.significand;
                const std::uint64_t corrected_magnitude = magnitude(transient);
                if (corrected_magnitude != 0) {
                    const int corrected_exponent = divisor_parts.unit_exponent + floor_log2(corrected_magnitude);
                    if (corrected_exponent < -126) {
                        return make_evaluation(
                            Outcome::IntermediateSubnormal,
                            config,
                            input_bits,
                            residual_bits,
                            expected_bits,
                            shift,
                            quotient_hat,
                            quotient_exact,
                            corrected_magnitude,
                            {
                                .stage_index = stage_index,
                                .exact_residual_exponent = corrected_exponent,
                                .subnormal_point = SubnormalPoint::AfterCorrection,
                                .negative_subnormal = transient < 0,
                            });
                    }
                }
            }
            if (transient < 0 || static_cast<std::uint64_t>(transient) != exact_remainder) {
                return make_evaluation(
                    Outcome::CorrectionMismatch,
                    config,
                    input_bits,
                    residual_bits,
                    expected_bits,
                    shift,
                    quotient_hat,
                    quotient_exact,
                    magnitude(transient));
            }

            const auto next_residual = compose_exact_bits(exact_remainder, divisor_parts.unit_exponent);
            if (!next_residual) {
                return make_evaluation(
                    Outcome::InternalInvariant,
                    config,
                    input_bits,
                    residual_bits,
                    expected_bits,
                    shift,
                    quotient_hat,
                    quotient_exact,
                    exact_remainder,
                    {
                        .stage_index = stage_index,
                        .exact_residual_exponent = divisor_parts.unit_exponent + floor_log2(exact_remainder),
                        .subnormal_point =
                            quotient_error == 1 ? SubnormalPoint::AfterCorrection : SubnormalPoint::AfterLowSubtract,
                    });
            }
            if (*next_residual != 0 && *next_residual < kPositiveNormalBegin) {
                return make_evaluation(
                    Outcome::ExactRemainderSubnormal,
                    config,
                    input_bits,
                    *next_residual,
                    expected_bits,
                    shift,
                    quotient_hat,
                    quotient_exact,
                    exact_remainder);
            }
            residual_bits = *next_residual;
        }

        if (shift == 0) {
            break;
        }
        const int decrement = std::min(kChunkStep, shift);
        shift -= decrement;
        if (config.exponent_stationary) {
            if (residual_bits != 0) {
                const auto next_residual = scale_normal_bits(residual_bits, decrement);
                if (!next_residual) {
                    return make_evaluation(
                        Outcome::IntermediateOverflow,
                        config,
                        input_bits,
                        residual_bits,
                        expected_bits,
                        shift,
                        0,
                        0,
                        residual_bits,
                        {
                            .stage_index = stage_index,
                        });
                }
                residual_bits = *next_residual;
            }
        } else {
            divisor_bits = scale_normal_bits(*divisor_bits, -decrement);
            const auto next_inverse = scale_normal_bits(inverse_bits, decrement);
            if (!divisor_bits) {
                return make_evaluation(
                    Outcome::ScaledDivisorNotNormal, config, input_bits, residual_bits, expected_bits, shift);
            }
            if (!next_inverse) {
                return make_evaluation(
                    Outcome::InverseNotNormal, config, input_bits, residual_bits, expected_bits, shift);
            }
            inverse_bits = *next_inverse;
        }
        ++stage_index;
    }

    if (prehalved && residual_bits != 0) {
        const auto doubled = scale_normal_bits(residual_bits, 1);
        if (!doubled) {
            return make_evaluation(Outcome::PostReductionOverflow, config, input_bits, residual_bits, expected_bits);
        }
        residual_bits = *doubled;
        const std::uint32_t reconstruction_divisor_bits =
            config.exponent_stationary ? config.reduction_divisor_bits : config.divisor_bits;
        if (residual_bits >= reconstruction_divisor_bits) {
            const auto corrected = exact_mod_bits(residual_bits, reconstruction_divisor_bits);
            if (!corrected) {
                return make_evaluation(Outcome::InternalInvariant, config, input_bits, residual_bits, expected_bits);
            }
            if (*corrected != 0 && *corrected < kPositiveNormalBegin) {
                return make_evaluation(
                    Outcome::ExactRemainderSubnormal,
                    config,
                    input_bits,
                    *corrected,
                    expected_bits,
                    0,
                    0,
                    0,
                    *corrected,
                    {
                        .stage_index = stage_index,
                        .exact_residual_exponent = -149 + floor_log2(*corrected),
                        .subnormal_point = SubnormalPoint::AfterCorrection,
                    });
            }
            residual_bits = *corrected;
        }
    }

    if (config.exponent_stationary) {
        const auto packed = pack_scaled_fp32_exact(residual_bits, config.final_exponent_shift);
        if (!packed) {
            return make_evaluation(Outcome::PostReductionOverflow, config, input_bits, residual_bits, expected_bits);
        }
        residual_bits = *packed;
    }

    if (residual_bits != expected_bits) {
        return make_evaluation(Outcome::FinalMismatch, config, input_bits, residual_bits, expected_bits);
    }
    EvaluationMetadata final_metadata;
    if (config.exponent_stationary && classify_bits(expected_bits) == ValueClass::Subnormal) {
        final_metadata.stage_index = stage_index;
        final_metadata.exact_residual_exponent = -149 + floor_log2(expected_bits);
        final_metadata.subnormal_point = SubnormalPoint::FinalResult;
    }
    return make_evaluation(
        Outcome::Passed, config, input_bits, residual_bits, expected_bits, -1, 0, 0, 0, final_metadata);
}

void write_records(const std::string& path, const std::vector<Record>& records) {
    if (path.empty()) {
        return;
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open record output: " + path);
    }
    output << "outcome\tinput_bits\tdivisor_bits\tgot_bits\texpected_bits\tstage_index\tstage_shift\t"
              "divisor_exponent\texact_residual_exponent\tq_hat\tq_exact\tq_error\tsubnormal_point\t"
              "final_exact_class\tnegative_subnormal\tdetail\n";
    output << std::setfill('0');
    for (const Record& record : records) {
        output << std::hex << kOutcomeNames[static_cast<std::size_t>(record.outcome)] << "\t0x" << std::setw(8)
               << record.input_bits << "\t0x" << std::setw(8) << record.divisor_bits << "\t0x" << std::setw(8)
               << record.got_bits << "\t0x" << std::setw(8) << record.expected_bits << std::dec << '\t'
               << record.stage_index << '\t' << record.stage_shift << '\t' << record.divisor_exponent << '\t'
               << record.exact_residual_exponent << '\t' << record.quotient_hat << '\t' << record.quotient_exact << '\t'
               << static_cast<std::int64_t>(record.quotient_hat) - record.quotient_exact << '\t'
               << kSubnormalPointNames[static_cast<std::size_t>(record.subnormal_point)] << '\t'
               << kValueClassNames[static_cast<std::size_t>(record.final_exact_class)] << '\t'
               << record.negative_subnormal << std::hex << "\t0x" << record.detail << std::dec << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        validate_exact_packer();
        const Options options = parse_options(argc, argv);
        const auto config_optional =
            make_config(options.divisor_bits, options.exponent_stationary, options.working_exponent);
        if (!config_optional) {
            std::cerr << "configuration_outcome DivisorOrNormalizedReciprocalNotNormal\n";
            return 2;
        }
        const Config config = *config_optional;

        const std::uint64_t total_span = options.input_end - options.input_begin;
        const std::uint64_t shard_begin = options.input_begin + (total_span * options.shard) / options.num_shards;
        const std::uint64_t shard_end = options.input_begin + (total_span * (options.shard + 1)) / options.num_shards;
        const std::uint64_t shard_span = shard_end - shard_begin;
        const unsigned worker_count = static_cast<unsigned>(std::max<std::uint64_t>(
            1, std::min<std::uint64_t>(options.threads, std::max<std::uint64_t>(1, shard_span))));

        std::cout << std::hex << std::setfill('0') << "divisor_bits 0x" << std::setw(8) << config.divisor_bits
                  << "\nreciprocal_up_bits 0x" << std::setw(8) << config.reciprocal_up_bits
                  << "\ninitial_inverse_bits 0x" << std::setw(8) << config.initial_inverse_bits << std::dec
                  << "\nreduction_divisor_bits 0x" << std::hex << std::setw(8) << config.reduction_divisor_bits
                  << std::dec << "\nstart_shift " << config.start_shift << "\nhigh_mantissa " << config.high_mantissa
                  << "\nexponent_stationary " << config.exponent_stationary << "\nworking_exponent "
                  << config.working_exponent << "\nfinal_exponent_shift " << config.final_exponent_shift
                  << "\nrange_begin 0x" << std::hex << shard_begin << "\nrange_end 0x" << shard_end << std::dec
                  << "\nshard " << options.shard << '/' << options.num_shards << "\nthreads " << worker_count
                  << "\nprehalve_max_exponent " << (options.prehalve_max_exponent || options.exponent_stationary)
                  << '\n';

        std::atomic<bool> stop = false;
        std::vector<ThreadResult> results(worker_count);
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        const auto start_time = std::chrono::steady_clock::now();

        for (unsigned worker = 0; worker < worker_count; ++worker) {
            const std::uint64_t begin = shard_begin + (shard_span * worker) / worker_count;
            const std::uint64_t end = shard_begin + (shard_span * (worker + 1)) / worker_count;
            workers.emplace_back([&, worker, begin, end] {
                ThreadResult& result = results[worker];
                result.failures.reserve(options.max_records);
                result.exclusions.reserve(options.max_records);
                for (std::uint64_t value = begin; value < end && !stop.load(std::memory_order_relaxed); ++value) {
                    const Evaluation evaluation =
                        evaluate(static_cast<std::uint32_t>(value), config, options.prehalve_max_exponent);
                    ++result.counts[static_cast<std::size_t>(evaluation.outcome)];
                    if (evaluation.record.subnormal_point != SubnormalPoint::None) {
                        const std::size_t point_index = static_cast<std::size_t>(evaluation.record.subnormal_point);
                        const std::size_t class_index = static_cast<std::size_t>(evaluation.record.final_exact_class);
                        ++result.subnormal_point_counts[point_index];
                        ++result.subnormal_final_class_counts[point_index][class_index];
                        result.negative_subnormal_count += evaluation.record.negative_subnormal;
                    }
                    if (evaluation.outcome != Outcome::Passed) {
                        std::vector<Record>& records =
                            is_failure(evaluation.outcome) ? result.failures : result.exclusions;
                        if (records.size() < options.max_records) {
                            records.push_back(evaluation.record);
                        }
                    }
                    if (options.stop_on_failure && is_failure(evaluation.outcome)) {
                        stop.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }

        std::array<std::uint64_t, static_cast<std::size_t>(Outcome::Count)> counts{};
        std::array<std::uint64_t, static_cast<std::size_t>(SubnormalPoint::Count)> subnormal_point_counts{};
        std::array<
            std::array<std::uint64_t, static_cast<std::size_t>(ValueClass::Count)>,
            static_cast<std::size_t>(SubnormalPoint::Count)>
            subnormal_final_class_counts{};
        std::uint64_t negative_subnormal_count = 0;
        std::vector<Record> failures;
        std::vector<Record> exclusions;
        for (const ThreadResult& result : results) {
            for (std::size_t index = 0; index < counts.size(); ++index) {
                counts[index] += result.counts[index];
            }
            for (std::size_t point = 0; point < subnormal_point_counts.size(); ++point) {
                subnormal_point_counts[point] += result.subnormal_point_counts[point];
                for (std::size_t value_class = 0; value_class < static_cast<std::size_t>(ValueClass::Count);
                     ++value_class) {
                    subnormal_final_class_counts[point][value_class] +=
                        result.subnormal_final_class_counts[point][value_class];
                }
            }
            negative_subnormal_count += result.negative_subnormal_count;
            failures.insert(failures.end(), result.failures.begin(), result.failures.end());
            exclusions.insert(exclusions.end(), result.exclusions.begin(), result.exclusions.end());
        }
        if (failures.size() > options.max_records) {
            failures.resize(options.max_records);
        }
        if (exclusions.size() > options.max_records) {
            exclusions.resize(options.max_records);
        }

        std::uint64_t processed = 0;
        std::uint64_t failure_count = 0;
        for (std::size_t index = 0; index < counts.size(); ++index) {
            if (counts[index] != 0) {
                std::cout << "outcome " << kOutcomeNames[index] << ' ' << counts[index] << '\n';
            }
            processed += counts[index];
            if (is_failure(static_cast<Outcome>(index))) {
                failure_count += counts[index];
            }
        }
        for (std::size_t point = 1; point < subnormal_point_counts.size(); ++point) {
            if (subnormal_point_counts[point] == 0) {
                continue;
            }
            std::cout << "subnormal_point " << kSubnormalPointNames[point] << ' ' << subnormal_point_counts[point]
                      << '\n';
            for (std::size_t value_class = 0; value_class < static_cast<std::size_t>(ValueClass::Count);
                 ++value_class) {
                if (subnormal_final_class_counts[point][value_class] != 0) {
                    std::cout << "subnormal_final_class " << kSubnormalPointNames[point] << ' '
                              << kValueClassNames[value_class] << ' '
                              << subnormal_final_class_counts[point][value_class] << '\n';
                }
            }
        }
        if (negative_subnormal_count != 0) {
            std::cout << "negative_subnormal_before_correction " << negative_subnormal_count << '\n';
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        std::cout << "processed " << processed << "\nfailures " << failure_count << "\nelapsed_seconds " << elapsed
                  << "\nrate_mvalues_s " << (elapsed == 0.0 ? 0.0 : processed / elapsed / 1.0e6) << '\n';

        write_records(options.dump_failures, failures);
        write_records(options.dump_exclusions, exclusions);
        return failure_count == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
