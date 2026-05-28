#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>

#define VC4_CASES 6u
#define VC4_TILE_WORDS 16u
#define VC4_GUARD_WORDS 7u
#define VC4_TOTAL_WORDS (VC4_GUARD_WORDS + VC4_CASES * VC4_TILE_WORDS + VC4_GUARD_WORDS)
#define VC4_SENTINEL 0x71710a04u

static const uint32_t k_active[VC4_CASES] = {0u, 1u, 5u, 9u, 13u, 16u};

static uint32_t input_word(uint32_t i) {
    return 11u + 7u * i;
}

static uint32_t expected_sum(uint32_t active) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < active; ++i)
        sum += input_word(i);
    return sum;
}

static uint32_t expected_word(uint32_t index) {
    if (index < VC4_GUARD_WORDS ||
        index >= VC4_GUARD_WORDS + VC4_CASES * VC4_TILE_WORDS)
        return VC4_SENTINEL;
    uint32_t logical = index - VC4_GUARD_WORDS;
    uint32_t which = logical / VC4_TILE_WORDS;
    return expected_sum(k_active[which]);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_in = 0, d_out = 0;
    uint32_t host_in[VC4_TILE_WORDS];
    uint32_t host_out[VC4_TOTAL_WORDS];
    uint32_t got[VC4_TOTAL_WORDS];
    uint32_t total_mismatches = 0;
    uint32_t sentinel_mismatches = 0;
    uint32_t checksum_actual = 0;
    uint32_t checksum_expected = 0;
    int failures = 0;

    for (uint32_t i = 0; i < VC4_TILE_WORDS; ++i)
        host_in[i] = input_word(i);
    for (uint32_t i = 0; i < VC4_TOTAL_WORDS; ++i) {
        host_out[i] = VC4_SENTINEL;
        got[i] = 0;
    }

    failures += vc4_m2_check_rc("vc4_program_create",
                                vc4_program_create(&program, 8192));
    if (!failures)
        failures += vc4_m2_malloc(program, &d_in, sizeof(host_in)) < 0;
    if (!failures)
        failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_in, host_in,
                                     sizeof(host_in)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_out, host_out,
                                     sizeof(host_out)) < 0;

    for (uint32_t c = 0; !failures && c < VC4_CASES; ++c) {
        vc4_deviceptr_t out_active =
            d_out + (VC4_GUARD_WORDS + c * VC4_TILE_WORDS) * sizeof(uint32_t);
        failures += vc4_m2_check_rc(
            "tile_reduce_tail_sum_vc4tile_launch",
            tile_reduce_tail_sum_vc4tile_launch(
                program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1),
                out_active, d_in, k_active[c]));
    }

    if (!failures)
        failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t i = 0; i < VC4_TOTAL_WORDS; ++i) {
        uint32_t expected = expected_word(i);
        checksum_actual += got[i];
        checksum_expected += expected;
        if (got[i] != expected) {
            if (expected == VC4_SENTINEL)
                ++sentinel_mismatches;
            else
                ++total_mismatches;
        }
    }

    uint32_t launch_failures =
        tile_reduce_tail_sum_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches = tile_reduce_tail_sum_vc4tile_runtime_launches();
    const char *status =
        (!failures && total_mismatches == 0 && sentinel_mismatches == 0 &&
         launch_failures == 0 && runtime_launches == VC4_CASES)
            ? "PASS"
            : "FAIL";
    printk("VC4_TEST_RESULT name=tile_reduce_tail_sum_vc4tile status=%s cases=%u predicated_reduce_path=1 total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u sample_n0_lane0=%u sample_n9_lane7=%u sample_n16_lane15=%u\n",
           status, VC4_CASES, total_mismatches, sentinel_mismatches,
           launch_failures, runtime_launches, checksum_actual,
           checksum_expected, got[VC4_GUARD_WORDS],
           got[VC4_GUARD_WORDS + 3u * VC4_TILE_WORDS + 7u],
           got[VC4_GUARD_WORDS + 5u * VC4_TILE_WORDS + 15u]);
    printk("DONE!!!\n");
}
