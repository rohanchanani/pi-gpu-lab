#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>
#include <string.h>

#define VC4_ROWS 32u
#define VC4_COLS 32u
#define VC4_ACTIVE_WORDS (VC4_ROWS * VC4_COLS)
#define VC4_BUFFER_WORDS 1048u
#define VC4_ACTIVE_OFFSET 11u
#define VC4_SENTINEL 0x9c9c5a5au

static uint32_t src_value(uint32_t row, uint32_t col) {
    return 1000u * row + col;
}

static uint32_t expected_value(uint32_t row, uint32_t col) {
    return src_value(col, row);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_in = 0, d_out = 0;
    uint32_t host_in[VC4_BUFFER_WORDS], host_out[VC4_BUFFER_WORDS], got[VC4_BUFFER_WORDS];
    uint32_t total_mismatches = 0, sentinel_mismatches = 0;
    uint32_t checksum_actual = 0, checksum_expected = 0;
    int failures = 0;

    for (uint32_t i = 0; i < VC4_BUFFER_WORDS; ++i) {
        host_in[i] = VC4_SENTINEL;
        host_out[i] = VC4_SENTINEL;
        got[i] = 0;
    }
    for (uint32_t r = 0; r < VC4_ROWS; ++r)
        for (uint32_t c = 0; c < VC4_COLS; ++c)
            host_in[VC4_ACTIVE_OFFSET + r * VC4_COLS + c] = src_value(r, c);

    failures += vc4_m2_check_rc("vc4_program_create", vc4_program_create(&program, 16384));
    if (!failures) failures += vc4_m2_malloc(program, &d_in, sizeof(host_in)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_in, host_in, sizeof(host_in)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_out, host_out, sizeof(host_out)) < 0;
    if (!failures) {
        vc4_deviceptr_t in_active = d_in + VC4_ACTIVE_OFFSET * sizeof(uint32_t);
        vc4_deviceptr_t out_active = d_out + VC4_ACTIVE_OFFSET * sizeof(uint32_t);
        failures += vc4_m2_check_rc(
            "cute_transpose_32x32_full_vpm_vc4tile_launch",
            cute_transpose_32x32_full_vpm_vc4tile_launch(
                program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1), out_active, in_active));
    }
    if (!failures) failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t r = 0; r < VC4_ROWS; ++r) {
        for (uint32_t c = 0; c < VC4_COLS; ++c) {
            uint32_t idx = VC4_ACTIVE_OFFSET + r * VC4_COLS + c;
            uint32_t expected = expected_value(r, c);
            checksum_actual += got[idx];
            checksum_expected += expected;
            if (got[idx] != expected)
                ++total_mismatches;
        }
    }
    for (uint32_t i = 0; i < VC4_BUFFER_WORDS; ++i) {
        uint32_t active = (i >= VC4_ACTIVE_OFFSET && i < VC4_ACTIVE_OFFSET + VC4_ACTIVE_WORDS);
        if (!active && got[i] != VC4_SENTINEL)
            ++sentinel_mismatches;
    }

    uint32_t launch_failures = cute_transpose_32x32_full_vpm_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches = cute_transpose_32x32_full_vpm_vc4tile_runtime_launches();
    const char *status = (!failures && total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && runtime_launches == 1) ? "PASS" : "FAIL";
    printk("shape=32x32 logical_bytes=4096\n");
    printk("samples dst00=%u dst031=%u dst173=%u dst310=%u dst3131=%u\n",
           got[VC4_ACTIVE_OFFSET + 0 * VC4_COLS + 0],
           got[VC4_ACTIVE_OFFSET + 0 * VC4_COLS + 31],
           got[VC4_ACTIVE_OFFSET + 17 * VC4_COLS + 3],
           got[VC4_ACTIVE_OFFSET + 31 * VC4_COLS + 0],
           got[VC4_ACTIVE_OFFSET + 31 * VC4_COLS + 31]);
    printk("VC4_TEST_RESULT name=cute_transpose_32x32_full_vpm_vc4tile status=%s shape=32x32 logical_bytes=4096 total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_expected=%u checksum_actual=%u sample_dst00=%u sample_dst031=%u sample_dst173=%u sample_dst310=%u sample_dst3131=%u\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           runtime_launches, checksum_expected, checksum_actual,
           got[VC4_ACTIVE_OFFSET + 0 * VC4_COLS + 0],
           got[VC4_ACTIVE_OFFSET + 0 * VC4_COLS + 31],
           got[VC4_ACTIVE_OFFSET + 17 * VC4_COLS + 3],
           got[VC4_ACTIVE_OFFSET + 31 * VC4_COLS + 0],
           got[VC4_ACTIVE_OFFSET + 31 * VC4_COLS + 31]);
    printk("DONE!!!\n");
}
