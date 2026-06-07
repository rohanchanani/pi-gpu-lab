#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ACTIVE_QPUS 8u
#define ACTIVE_N (LANES * ACTIVE_QPUS)
#define GUARD 16u
#define BUFFER_N (ACTIVE_N + GUARD)
#define TAG 0x65000000u

static uint32_t out_values[BUFFER_N];

static uint32_t sentinel_value(uint32_t index) {
    return 0xcafe0000u + index;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = sentinel_value(i);
}

static uint32_t expected_value(uint32_t request, uint32_t stride,
                               uint32_t bias, uint32_t lane) {
    uint32_t amount = request * stride + bias;
    uint32_t source_lane = (lane + (amount & 15u)) & 15u;
    return TAG + request * LANES + source_lane;
}

static int verify_active(uint32_t stride, uint32_t bias) {
    int mismatches = 0;
    for (uint32_t request = 0; request < ACTIVE_QPUS; ++request) {
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t index = request * LANES + lane;
            uint32_t expected = expected_value(request, stride, bias, lane);
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: dynamic_rotate_amount_sources request=%d lane=%d gpu=%x expected=%x\n",
                           (int)request, (int)lane, out_values[index],
                           expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = ACTIVE_N; i < BUFFER_N; ++i) {
        uint32_t expected = sentinel_value(i);
        if (out_values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: dynamic_rotate_amount_sources sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], expected);
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
    uint32_t bytes = BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("fragment_rotate_dynamic_amount_sources allocation failed");

    const uint32_t stride = 5u;
    const uint32_t bias = 3u;
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(ACTIVE_QPUS, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    fill_output();
    printk("Running VC4 fragment_rotate_dynamic_amount_sources_vc4kernel candidate bundle...\n");
    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        fragment_rotate_dynamic_amount_sources_vc4kernel_launch(
            program, grid, block, out_dev, stride, bias) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
        printk("ERROR: fragment_rotate_dynamic_amount_sources launch/copy failed\n");
        launch_failures++;
    }

    int total_mismatches = launch_failures ? 0 : verify_active(stride, bias);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int checksum = launch_failures ? 0 : checksum_low16();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_rotate_dynamic_amount_sources_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d saw_dynamic_rotate=1 saw_amount_source_program_id=1 saw_amount_source_scalar_arith=1 saw_amount16_or_modulo=1 saw_i32=1 saw_vdw_preserve=1 rotate_direction=left_source_plus_amount checksum_accum=%d runtime_allocations=1 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)ACTIVE_QPUS, (int)LANES, checksum, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
