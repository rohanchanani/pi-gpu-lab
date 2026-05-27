#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>
#include <string.h>

#define VC4_WORDS 48u
#define VC4_ACTIVE_OFFSET 4u
#define VC4_ACTIVE_WORDS 32u
#define VC4_SENTINEL 0x71710005u

static uint32_t x_value(uint32_t i) {
    return 19u + 5u * i;
}

static uint32_t y_value(uint32_t i) {
    return 0x1100u + 7u * i;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_x = 0, d_y = 0, d_out = 0;
    uint32_t host_x[VC4_WORDS], host_y[VC4_WORDS], host_out[VC4_WORDS], got[VC4_WORDS];
    uint32_t total_mismatches = 0, sentinel_mismatches = 0;
    uint32_t checksum_actual = 0, checksum_expected = 0;
    int failures = 0;

    for (uint32_t i = 0; i < VC4_WORDS; ++i) {
        host_x[i] = VC4_SENTINEL;
        host_y[i] = VC4_SENTINEL;
        host_out[i] = VC4_SENTINEL;
        got[i] = 0;
    }
    for (uint32_t i = 0; i < VC4_ACTIVE_WORDS; ++i) {
        host_x[VC4_ACTIVE_OFFSET + i] = x_value(i);
        host_y[VC4_ACTIVE_OFFSET + i] = y_value(i);
    }

    failures += vc4_m2_check_rc("vc4_program_create", vc4_program_create(&program, 8192));
    if (!failures) failures += vc4_m2_malloc(program, &d_x, sizeof(host_x)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_y, sizeof(host_y)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_x, host_x, sizeof(host_x)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_y, host_y, sizeof(host_y)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_out, host_out, sizeof(host_out)) < 0;
    if (!failures) {
        vc4_deviceptr_t x_active = d_x + VC4_ACTIVE_OFFSET * sizeof(uint32_t);
        vc4_deviceptr_t y_active = d_y + VC4_ACTIVE_OFFSET * sizeof(uint32_t);
        vc4_deviceptr_t out_active = d_out + VC4_ACTIVE_OFFSET * sizeof(uint32_t);
        failures += vc4_m2_check_rc(
            "scf_tiled_saxpy_loop_vc4tile_launch",
            scf_tiled_saxpy_loop_vc4tile_launch(
                program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1),
                out_active, x_active, y_active, VC4_ACTIVE_WORDS));
    }
    if (!failures) failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t i = 0; i < VC4_WORDS; ++i) {
        uint32_t active = (i >= VC4_ACTIVE_OFFSET && i < VC4_ACTIVE_OFFSET + VC4_ACTIVE_WORDS);
        uint32_t lane = i - VC4_ACTIVE_OFFSET;
        uint32_t expected = active ? (2u * x_value(lane) + y_value(lane)) : VC4_SENTINEL;
        checksum_actual += got[i];
        checksum_expected += expected;
        if (got[i] != expected) {
            if (active) ++total_mismatches;
            else ++sentinel_mismatches;
        }
    }

    uint32_t launch_failures = scf_tiled_saxpy_loop_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches = scf_tiled_saxpy_loop_vc4tile_runtime_launches();
    const char *status = (!failures && total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && runtime_launches == 1) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=scf_tiled_saxpy_loop_vc4tile status=%s total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           runtime_launches, checksum_actual, checksum_expected);
    printk("DONE!!!\n");
}
