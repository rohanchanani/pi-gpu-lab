#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>

#define VC4_CASES 6u
#define VC4_SOURCE_WORDS 16u
#define VC4_OUTPUT_WORDS 64u
#define VC4_GUARD_WORDS 9u
#define VC4_TOTAL_WORDS (VC4_GUARD_WORDS + VC4_CASES * VC4_OUTPUT_WORDS + VC4_GUARD_WORDS)
#define VC4_SENTINEL 0x5a5ac041u

static const uint32_t k_rows[VC4_CASES] = {1u, 1u, 2u, 3u, 4u, 4u};
static const uint32_t k_cols[VC4_CASES] = {1u, 4u, 3u, 2u, 1u, 4u};

static uint32_t input_word(uint32_t which, uint32_t lane) {
    return 0x51000000u + which * 263u + lane * 19u;
}

static uint32_t expected_row_word(uint32_t which, uint32_t row, uint32_t lane) {
    if (row < k_rows[which] && lane < k_cols[which])
        return input_word(which, row * 4u + lane);
    return 0u;
}

static uint32_t expected_word(uint32_t index) {
    if (index < VC4_GUARD_WORDS ||
        index >= VC4_GUARD_WORDS + VC4_CASES * VC4_OUTPUT_WORDS)
        return VC4_SENTINEL;

    uint32_t logical = index - VC4_GUARD_WORDS;
    uint32_t which = logical / VC4_OUTPUT_WORDS;
    uint32_t lane = logical % VC4_OUTPUT_WORDS;
    return expected_row_word(which, lane / 16u, lane % 16u);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_out = 0;
    vc4_deviceptr_t d_in = 0;
    uint32_t host_out[VC4_TOTAL_WORDS];
    uint32_t host_in[VC4_CASES * VC4_SOURCE_WORDS];
    uint32_t got[VC4_TOTAL_WORDS];
    uint32_t total_mismatches = 0;
    uint32_t sentinel_mismatches = 0;
    uint32_t checksum_actual = 0;
    uint32_t checksum_expected = 0;
    int failures = 0;

    for (uint32_t i = 0; i < VC4_TOTAL_WORDS; ++i) {
        host_out[i] = VC4_SENTINEL;
        got[i] = 0;
    }
    for (uint32_t c = 0; c < VC4_CASES; ++c) {
        for (uint32_t lane = 0; lane < VC4_SOURCE_WORDS; ++lane)
            host_in[c * VC4_SOURCE_WORDS + lane] = input_word(c, lane);
    }

    failures += vc4_m2_check_rc("vc4_program_create",
                                vc4_program_create(&program, 8192));
    if (!failures)
        failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures)
        failures += vc4_m2_malloc(program, &d_in, sizeof(host_in)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_out, host_out,
                                     sizeof(host_out)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_in, host_in,
                                     sizeof(host_in)) < 0;

    for (uint32_t c = 0; !failures && c < VC4_CASES; ++c) {
        vc4_deviceptr_t out_active =
            d_out + (VC4_GUARD_WORDS + c * VC4_OUTPUT_WORDS) * sizeof(uint32_t);
        vc4_deviceptr_t in_active =
            d_in + c * VC4_SOURCE_WORDS * sizeof(uint32_t);
        failures += vc4_m2_check_rc(
            "tile_bounds_mask_global_shared_load_vc4tile_launch",
            tile_bounds_mask_global_shared_load_vc4tile_launch(
                program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1),
                out_active, in_active, k_rows[c], k_cols[c]));
    }

    if (!failures)
        failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t i = 0; i < VC4_TOTAL_WORDS; ++i) {
        uint32_t expected = expected_word(i);
        checksum_actual += got[i];
        checksum_expected += expected;
        if (got[i] != expected) {
            if (expected == VC4_SENTINEL) {
                ++sentinel_mismatches;
            } else {
                if (total_mismatches < 16u) {
                    uint32_t logical = i - VC4_GUARD_WORDS;
                    uint32_t which = logical / VC4_OUTPUT_WORDS;
                    uint32_t lane = logical % VC4_OUTPUT_WORDS;
                    printk("mismatch case=%u row=%u lane=%u got=%u expected=%u\n",
                           which, lane / 16u, lane % 16u, got[i], expected);
                }
                ++total_mismatches;
            }
        }
    }

    uint32_t launch_failures =
        tile_bounds_mask_global_shared_load_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches =
        tile_bounds_mask_global_shared_load_vc4tile_runtime_launches();
    const char *status =
        (!failures && total_mismatches == 0 && sentinel_mismatches == 0 &&
         launch_failures == 0 && runtime_launches == VC4_CASES)
            ? "PASS"
            : "FAIL";
    printk("VC4_TEST_RESULT name=tile_bounds_mask_global_shared_load_vc4tile status=%s cases=%u shared_predicated_path=1 total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u sample_1x1_row0_lane0=%u sample_3x2_row2_lane1=%u sample_4x4_row3_lane3=%u sample_inactive_row3_lane0=%u\n",
           status, VC4_CASES, total_mismatches, sentinel_mismatches,
           launch_failures, runtime_launches, checksum_actual,
           checksum_expected,
           got[VC4_GUARD_WORDS],
           got[VC4_GUARD_WORDS + 3u * VC4_OUTPUT_WORDS + 2u * 16u + 1u],
           got[VC4_GUARD_WORDS + 5u * VC4_OUTPUT_WORDS + 3u * 16u + 3u],
           got[VC4_GUARD_WORDS + 0u * VC4_OUTPUT_WORDS + 3u * 16u]);
    printk("DONE!!!\n");
}
