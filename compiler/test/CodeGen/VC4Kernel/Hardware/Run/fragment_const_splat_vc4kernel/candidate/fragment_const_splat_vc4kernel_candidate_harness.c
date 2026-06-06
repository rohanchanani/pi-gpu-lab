#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 7u
#define ACTIVE_N (LANES * SEGMENTS)
#define BUFFER_N (ACTIVE_N + 16u)
#define SENTINEL 0xdeadbeefu

static uint32_t out_values[BUFFER_N];

static uint32_t expected_value(uint32_t segment) {
    static const uint32_t values[SEGMENTS] = {
        0x00000000u, 0xffffffffu, 0xf8a432ebu, 0x00000000u,
        0x80000000u, 0x3f800000u, 0xc0200000u
    };
    return values[segment];
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
        uint32_t expected = expected_value(segment);
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t index = segment * LANES + lane;
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: fragment_const_splat segment=%d lane=%d gpu=%x expected=%x\n",
                           (int)segment, (int)lane, out_values[index], expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = ACTIVE_N; i < BUFFER_N; ++i) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_const_splat sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < ACTIVE_N; ++i)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        panic("fragment_const_splat_vc4kernel allocation failed");

    fill_output();
    int start = timer_get_usec();
    int launch_failures = 0;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, out_dev, out_values, BUFFER_N * sizeof(uint32_t)) < 0 ||
        fragment_const_splat_vc4kernel_launch(program, grid, block, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        launch_failures++;

    int total_mismatches = launch_failures ? 0 : verify_active();
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int checksum_accum = launch_failures ? 0 : checksum_low16();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_const_splat_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d segments=%d i32_splats=3 f32_splat_bitcasts=4 checksum_accum=%d runtime_allocations=1 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)SEGMENTS, checksum_accum, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
