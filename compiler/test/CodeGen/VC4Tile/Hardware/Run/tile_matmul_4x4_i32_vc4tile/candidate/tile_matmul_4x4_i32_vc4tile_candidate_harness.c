#include "vc4_m2_candidate_test_helpers.h"
#include <stdint.h>

#define VC4_WORDS 32u
#define VC4_SENTINEL 0x71110002u
#define VC4_ACC_VALUE 13u

static uint32_t lhs_value(uint32_t i) {
    uint32_t row = i / 4u;
    uint32_t kk = i % 4u;
    return 2u + 5u * row + kk;
}

static uint32_t rhs_value(uint32_t i) {
    uint32_t kk = i / 4u;
    uint32_t col = i % 4u;
    return 7u + 3u * kk + 11u * col;
}

static uint32_t matmul_expected(uint32_t row, uint32_t col) {
    uint32_t expected = VC4_ACC_VALUE;
    for (uint32_t kk = 0; kk < 4u; ++kk)
        expected += lhs_value(row * 4u + kk) * rhs_value(kk * 4u + col);
    return expected;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t d_out = 0, d_lhs = 0, d_rhs = 0;
    uint32_t host_lhs[VC4_WORDS], host_rhs[VC4_WORDS], host_out[VC4_WORDS], got[VC4_WORDS];
    uint32_t total_mismatches = 0, sentinel_mismatches = 0;
    uint32_t checksum_actual = 0, checksum_expected = 0;
    int failures = 0;

    for (uint32_t i = 0; i < VC4_WORDS; ++i) {
        host_lhs[i] = lhs_value(i);
        host_rhs[i] = rhs_value(i);
        host_out[i] = VC4_SENTINEL;
        got[i] = 0;
    }

    failures += vc4_m2_check_rc("vc4_program_create", vc4_program_create(&program, 8192));
    if (!failures) failures += vc4_m2_malloc(program, &d_out, sizeof(host_out)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_lhs, sizeof(host_lhs)) < 0;
    if (!failures) failures += vc4_m2_malloc(program, &d_rhs, sizeof(host_rhs)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_out, host_out, sizeof(host_out)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_lhs, host_lhs, sizeof(host_lhs)) < 0;
    if (!failures) failures += vc4_m2_copy_htod(program, d_rhs, host_rhs, sizeof(host_rhs)) < 0;
    if (!failures) {
        failures += vc4_m2_check_rc(
            "tile_matmul_4x4_i32_vc4tile_launch",
            tile_matmul_4x4_i32_vc4tile_launch(program, vc4_m2_dim3(1, 1, 1), vc4_m2_dim3(16, 1, 1),
                          d_out, d_lhs, d_rhs));
    }
    if (!failures) failures += vc4_m2_copy_dtoh(program, got, d_out, sizeof(got)) < 0;

    for (uint32_t i = 0; i < VC4_WORDS; ++i) {
        uint32_t expected = VC4_SENTINEL;
        if (i < 16u) {
            uint32_t row = i / 4u;
            uint32_t col = i % 4u;
            expected = matmul_expected(row, col);
        }
        checksum_actual += got[i];
        checksum_expected += expected;
        if (got[i] != expected) {
            if (i < 16u) ++total_mismatches;
            else ++sentinel_mismatches;
        }
    }

    uint32_t launch_failures = tile_matmul_4x4_i32_vc4tile_runtime_launch_failures();
    uint32_t runtime_launches = tile_matmul_4x4_i32_vc4tile_runtime_launches();
    const char *status = (!failures && total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && runtime_launches == 1) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=tile_matmul_4x4_i32_vc4tile status=%s total_mismatches=%u sentinel_mismatches=%u launch_failures=%u runtime_launches=%u checksum_actual=%u checksum_expected=%u\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           runtime_launches, checksum_actual, checksum_expected);
    printk("DONE!!!\n");
}
