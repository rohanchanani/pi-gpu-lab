#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>
#include <string.h>

#define VC4_ROWS 16u
#define VC4_COLS 16u
#define VC4_OUT_WORDS 16u
#define VC4_IN_WORDS 272u
#define VC4_OUT_BUFFER_WORDS 32u
#define VC4_IN_ACTIVE_OFFSET 3u
#define VC4_OUT_ACTIVE_OFFSET 5u
#define VC4_SENTINEL 0x8585d8d8u

static uint32_t matrix_value(uint32_t row, uint32_t col) {
    return 0x81000000u + 541u * row + 31u * col + (col << 3);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_in = 0, d_out = 0;
    uint32_t host_in[VC4_IN_WORDS], host_out[VC4_OUT_BUFFER_WORDS], got[VC4_OUT_BUFFER_WORDS];
    uint32_t total_mismatches = 0, sentinel_mismatches = 0;
    uint32_t checksum_actual = 0, checksum_expected = 0;
    int failures = 0;

    for (uint32_t i = 0; i < VC4_IN_WORDS; ++i) host_in[i] = VC4_SENTINEL;
    for (uint32_t i = 0; i < VC4_OUT_BUFFER_WORDS; ++i) {
        host_out[i] = VC4_SENTINEL;
        got[i] = 0;
    }
    for (uint32_t r = 0; r < VC4_ROWS; ++r)
        for (uint32_t c = 0; c < VC4_COLS; ++c)
            host_in[VC4_IN_ACTIVE_OFFSET + r * VC4_COLS + c] = matrix_value(r, c);

    failures += vc4_m2_check_rc("vc4_program_create", vc4_program_create(&program, 16384));
    if (!failures) failures += vc4_m2_malloc(program, &d_in, sizeof(host_in)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_in, host_in, sizeof(host_in)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_out, host_out, sizeof(host_out)) < 0;
    if (!failures) {
        vc4_deviceptr_t in_active = d_in + VC4_IN_ACTIVE_OFFSET * sizeof(uint32_t);
        vc4_deviceptr_t out_active = d_out + VC4_OUT_ACTIVE_OFFSET * sizeof(uint32_t);
        failures += vc4_m2_check_rc(
            "shared_transpose_store_global_vc4tile_launch",
            shared_transpose_store_global_vc4tile_launch(
                program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1), out_active, in_active));
    }
    if (!failures) failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t i = 0; i < VC4_OUT_BUFFER_WORDS; ++i) {
        uint32_t active = (i >= VC4_OUT_ACTIVE_OFFSET && i < VC4_OUT_ACTIVE_OFFSET + VC4_OUT_WORDS);
        uint32_t expected = active ? matrix_value(i - VC4_OUT_ACTIVE_OFFSET, 0) : VC4_SENTINEL;
        checksum_actual += got[i];
        checksum_expected += expected;
        if (got[i] != expected) {
            if (active) ++total_mismatches;
            else ++sentinel_mismatches;
        }
    }

    uint32_t launch_failures = shared_transpose_store_global_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches = shared_transpose_store_global_vc4tile_runtime_launches();
    const char *status = (!failures && total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && runtime_launches == 1) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=shared_transpose_store_global_vc4tile status=%s total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           runtime_launches, checksum_actual, checksum_expected);
    printk("DONE!!!\n");
}
