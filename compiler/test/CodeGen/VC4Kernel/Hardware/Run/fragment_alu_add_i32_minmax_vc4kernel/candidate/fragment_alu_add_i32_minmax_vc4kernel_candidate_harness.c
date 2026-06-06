#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 2u
#define BUFFER_N (LANES * SEGMENTS + 16u)
#define SENTINEL 0xdeadbeefu

static const int32_t a_values[LANES] = {
    -16, -1, 0, 1, 7, -128, 128, 1024,
    -1024, 2147483647, -2147483647 - 1, 42, 42, -9, 30000, -30000
};
static const int32_t b_values[LANES] = {
    -8, 1, 0, -1, 7, 127, -127, -2048,
    2048, -1, 0, 42, -42, -9, -30000, 30000
};
static uint32_t out_values[BUFFER_N];

static uint32_t expected_value(uint32_t segment, uint32_t lane) {
    int32_t a = a_values[lane];
    int32_t b = b_values[lane];
    int32_t value = segment == 0 ? (a < b ? a : b) : (a > b ? a : b);
    return (uint32_t)value;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t index = segment * LANES + lane;
            uint32_t expected = expected_value(segment, lane);
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: fragment_alu_add_i32_minmax seg=%d lane=%d gpu=%x expected=%x\n",
                           (int)segment, (int)lane, out_values[index], expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = SEGMENTS * LANES; i < BUFFER_N; ++i) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_alu_add_i32_minmax sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t a_dev = 0, b_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &b_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        panic("fragment_alu_add_i32_minmax allocation failed");

    fill_output();
    int start = timer_get_usec();
    int launch_failures = 0;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, a_dev, a_values, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, b_dev, b_values, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, BUFFER_N * sizeof(uint32_t)) < 0 ||
        fragment_alu_add_i32_minmax_vc4kernel_launch(program, grid, block, a_dev, b_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        launch_failures++;

    int total_mismatches = launch_failures ? 0 : verify_active();
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_alu_add_i32_minmax_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d op_segments=%d signed_i32_minmax=1 runtime_allocations=3 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)SEGMENTS, elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
