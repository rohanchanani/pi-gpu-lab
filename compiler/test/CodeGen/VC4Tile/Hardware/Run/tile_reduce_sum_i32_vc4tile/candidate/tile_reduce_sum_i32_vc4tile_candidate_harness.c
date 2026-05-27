#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>

#define VC4_WORDS 32u
#define VC4_ACTIVE_WORDS 16u
#define VC4_SENTINEL 0x71710a01u

static uint32_t x_value(uint32_t i) { return 7u + 3u * i; }

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_in = 0, d_out = 0;
    uint32_t host_in[VC4_WORDS], host_out[VC4_WORDS], got[VC4_WORDS];
    uint32_t total_mismatches = 0, sentinel_mismatches = 0;
    uint32_t checksum_actual = 0, checksum_expected = 0;
    uint32_t expected_sum = 0;
    int failures = 0;

    for (uint32_t i = 0; i < 16u; ++i)
        expected_sum += x_value(i);

    for (uint32_t i = 0; i < VC4_WORDS; ++i) {
        host_in[i] = x_value(i);
        host_out[i] = VC4_SENTINEL;
        got[i] = 0;
    }

    failures += vc4_m2_check_rc("vc4_program_create", vc4_program_create(&program, 8192));
    if (!failures) failures += vc4_m2_malloc(program, &d_in, sizeof(host_in)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_in, host_in, sizeof(host_in)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_out, host_out, sizeof(host_out)) < 0;
    if (!failures) {
        failures += vc4_m2_check_rc(
            "tile_reduce_sum_i32_vc4tile_launch",
            tile_reduce_sum_i32_vc4tile_launch(
                program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1),
                d_out, d_in));
    }
    if (!failures) failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t i = 0; i < VC4_WORDS; ++i) {
        uint32_t active = (i < 16u);
        uint32_t expected = active ? expected_sum : VC4_SENTINEL;
        checksum_actual += got[i];
        checksum_expected += expected;
        if (got[i] != expected) {
            if (active) ++total_mismatches;
            else ++sentinel_mismatches;
        }
    }

    uint32_t launch_failures = tile_reduce_sum_i32_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches = tile_reduce_sum_i32_vc4tile_runtime_launches();
    const char *status = (!failures && total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && runtime_launches == 1) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=tile_reduce_sum_i32_vc4tile status=%s total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           runtime_launches, checksum_actual, checksum_expected);
    printk("DONE!!!\n");
}
