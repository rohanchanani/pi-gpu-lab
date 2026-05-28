#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>

#define VC4_TILE_WORDS 16u
#define VC4_CASES 6u
#define VC4_GUARD_WORDS 8u
#define VC4_TOTAL_WORDS (VC4_GUARD_WORDS + VC4_CASES * VC4_TILE_WORDS + VC4_GUARD_WORDS)
#define VC4_SENTINEL 0x6d0d7002u

static const uint32_t k_rows[VC4_CASES] = {0u, 1u, 2u, 3u, 4u, 4u};
static const uint32_t k_cols[VC4_CASES] = {4u, 4u, 2u, 3u, 4u, 1u};
static const uint32_t k_tail[VC4_CASES] = {8u, 1u, 7u, 9u, 16u, 13u};

static uint32_t lhs_value(uint32_t lane) {
    return 5u + lane * 7u;
}

static uint32_t rhs_value(uint32_t lane) {
    return 19u + lane * 13u;
}

static uint32_t dot_expected(uint32_t rows, uint32_t cols, uint32_t tail) {
    uint32_t sum = 0;
    for (uint32_t lane = 0; lane < VC4_TILE_WORDS; ++lane) {
        uint32_t row = lane / 4u;
        uint32_t col = lane % 4u;
        if (row < rows && col < cols && lane < tail)
            sum += lhs_value(lane) * rhs_value(lane);
    }
    return sum;
}

static uint32_t expected_word(uint32_t index) {
    if (index < VC4_GUARD_WORDS)
        return VC4_SENTINEL;
    uint32_t logical = index - VC4_GUARD_WORDS;
    if (logical >= VC4_CASES * VC4_TILE_WORDS)
        return VC4_SENTINEL;
    uint32_t case_id = logical / VC4_TILE_WORDS;
    return dot_expected(k_rows[case_id], k_cols[case_id], k_tail[case_id]);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_out = 0, d_lhs = 0, d_rhs = 0;
    uint32_t host_lhs[VC4_TILE_WORDS];
    uint32_t host_rhs[VC4_TILE_WORDS];
    uint32_t host_out[VC4_TOTAL_WORDS];
    uint32_t got[VC4_TOTAL_WORDS];
    uint32_t total_mismatches = 0;
    uint32_t sentinel_mismatches = 0;
    uint32_t checksum_actual = 0;
    uint32_t checksum_expected = 0;
    int failures = 0;

    for (uint32_t lane = 0; lane < VC4_TILE_WORDS; ++lane) {
        host_lhs[lane] = lhs_value(lane);
        host_rhs[lane] = rhs_value(lane);
    }
    for (uint32_t i = 0; i < VC4_TOTAL_WORDS; ++i) {
        host_out[i] = VC4_SENTINEL;
        got[i] = 0;
    }

    failures += vc4_m2_check_rc("vc4_program_create",
                                vc4_program_create(&program, 8192));
    if (!failures)
        failures += vc4_m2_malloc(program, &d_lhs, sizeof(host_lhs)) < 0;
    if (!failures)
        failures += vc4_m2_malloc(program, &d_rhs, sizeof(host_rhs)) < 0;
    if (!failures)
        failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_lhs, host_lhs,
                                     sizeof(host_lhs)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_rhs, host_rhs,
                                     sizeof(host_rhs)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_out, host_out,
                                     sizeof(host_out)) < 0;

    for (uint32_t c = 0; !failures && c < VC4_CASES; ++c) {
        vc4_deviceptr_t out_active =
            d_out + (VC4_GUARD_WORDS + c * VC4_TILE_WORDS) * sizeof(uint32_t);
        failures += vc4_m2_check_rc(
            "tile_dot_composed_mask_vc4tile_launch",
            tile_dot_composed_mask_vc4tile_launch(
                program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1),
                out_active, d_lhs, d_rhs, k_rows[c], k_cols[c], k_tail[c]));
    }

    if (!failures)
        failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t i = 0; i < VC4_TOTAL_WORDS; ++i) {
        uint32_t expected = expected_word(i);
        checksum_actual += got[i];
        checksum_expected += expected;
        if (got[i] != expected) {
            if (i < VC4_GUARD_WORDS ||
                i >= VC4_GUARD_WORDS + VC4_CASES * VC4_TILE_WORDS)
                ++sentinel_mismatches;
            else
                ++total_mismatches;
        }
    }

    uint32_t launch_failures =
        tile_dot_composed_mask_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches =
        tile_dot_composed_mask_vc4tile_runtime_launches();
    const char *status = (!failures && total_mismatches == 0 &&
                          sentinel_mismatches == 0 && launch_failures == 0 &&
                          runtime_launches == VC4_CASES) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=tile_dot_composed_mask_vc4tile status=%s cases=%u composed_dot_predicate_path=1 total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u sample_0x4_tail8=%u sample_3x3_tail9=%u sample_4x1_tail13=%u\n",
           status, VC4_CASES, total_mismatches, sentinel_mismatches,
           launch_failures, runtime_launches, checksum_actual,
           checksum_expected, got[VC4_GUARD_WORDS],
           got[VC4_GUARD_WORDS + 3u * VC4_TILE_WORDS],
           got[VC4_GUARD_WORDS + 5u * VC4_TILE_WORDS]);
    printk("DONE!!!\n");
}
