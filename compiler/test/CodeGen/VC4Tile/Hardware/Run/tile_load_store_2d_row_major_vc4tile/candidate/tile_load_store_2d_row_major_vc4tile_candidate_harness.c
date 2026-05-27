#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>
#include <string.h>

#define VC4_MATRIX_ROWS 2u
#define VC4_MATRIX_COLS 16u
#define VC4_MATRIX_WORDS (VC4_MATRIX_ROWS * VC4_MATRIX_COLS)
#define VC4_CASE_WORDS 40u
#define VC4_CASE_ACTIVE_OFFSET 3u
#define VC4_CASE_SENTINEL 0x22440033u

static uint32_t matrix_value(uint32_t row, uint32_t col) {
    return 0x30000000u + 1009u * row + 13u * col + (row << 8);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_in = 0, d_out = 0;
    uint32_t host_in[VC4_CASE_WORDS];
    uint32_t host_out[VC4_CASE_WORDS];
    uint32_t got[VC4_CASE_WORDS];
    uint32_t total_mismatches = 0;
    uint32_t sentinel_mismatches = 0;
    uint32_t checksum_actual = 0;
    uint32_t checksum_expected = 0;
    int failures = 0;

    for (uint32_t i = 0; i < VC4_CASE_WORDS; ++i) {
        host_in[i] = VC4_CASE_SENTINEL;
        host_out[i] = VC4_CASE_SENTINEL;
        got[i] = 0;
    }
    for (uint32_t row = 0; row < VC4_MATRIX_ROWS; ++row) {
        for (uint32_t col = 0; col < VC4_MATRIX_COLS; ++col)
            host_in[VC4_CASE_ACTIVE_OFFSET + row * VC4_MATRIX_COLS + col] =
                matrix_value(row, col);
    }

    failures += vc4_m2_check_rc("vc4_program_create", vc4_program_create(&program, 8192));
    if (!failures)
        failures += vc4_m2_malloc(program, &d_in, sizeof(host_in)) < 0;
    if (!failures)
        failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_in, host_in, sizeof(host_in)) < 0;
    if (!failures)
        failures += vc4_m2_copy_htod(program, d_out, host_out, sizeof(host_out)) < 0;
    if (!failures) {
        vc4_deviceptr_t in_active = d_in + VC4_CASE_ACTIVE_OFFSET * sizeof(uint32_t);
        vc4_deviceptr_t out_active = d_out + VC4_CASE_ACTIVE_OFFSET * sizeof(uint32_t);
        failures += vc4_m2_check_rc(
            "tile_load_store_2d_row_major_vc4tile_launch",
            tile_load_store_2d_row_major_vc4tile_launch(
                program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1),
                out_active, in_active));
    }
    if (!failures)
        failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t i = 0; i < VC4_CASE_WORDS; ++i) {
        uint32_t active = (i >= VC4_CASE_ACTIVE_OFFSET &&
                           i < VC4_CASE_ACTIVE_OFFSET + VC4_MATRIX_WORDS);
        uint32_t expected = VC4_CASE_SENTINEL;
        if (active) {
            uint32_t linear = i - VC4_CASE_ACTIVE_OFFSET;
            expected = matrix_value(linear / VC4_MATRIX_COLS,
                                    linear % VC4_MATRIX_COLS);
        }
        checksum_actual += got[i];
        checksum_expected += expected;
        if (got[i] != expected) {
            if (active)
                ++total_mismatches;
            else
                ++sentinel_mismatches;
        }
    }

    uint32_t launch_failures = tile_load_store_2d_row_major_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches = tile_load_store_2d_row_major_vc4tile_runtime_launches();
    const char *status = (!failures && total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && runtime_launches == 1) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=tile_load_store_2d_row_major_vc4tile status=%s total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           runtime_launches, checksum_actual, checksum_expected);
    printk("DONE!!!\n");
}
