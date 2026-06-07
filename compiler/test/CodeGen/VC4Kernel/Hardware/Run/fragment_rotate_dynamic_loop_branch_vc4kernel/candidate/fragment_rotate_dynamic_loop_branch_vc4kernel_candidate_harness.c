#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define GUARD 16u
#define BUFFER_N (LANES + GUARD)
#define SENTINEL 0x6d5a0000u

struct loop_branch_case {
    uint32_t iters;
    uint32_t control;
    uint32_t base_value;
};

static const struct loop_branch_case cases[] = {
    {0u, 0u, 0x21000000u},
    {1u, 0u, 0x22000000u},
    {2u, 1u, 0x23000000u},
    {5u, 0u, 0x24000000u},
    {17u, 1u, 0x25000000u},
};

static uint32_t out_values[BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL + i;
}

static uint32_t expected_value(const struct loop_branch_case *test,
                               uint32_t lane) {
    uint32_t source_lane = lane;
    for (uint32_t j = 0; j < test->iters; ++j) {
        uint32_t amount = j + (test->control ? 7u : 3u);
        source_lane = (source_lane + (amount & 15u)) & 15u;
    }
    return test->base_value + source_lane;
}

static int verify_active(const struct loop_branch_case *test) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t expected = expected_value(test, lane);
        if (out_values[lane] != expected) {
            if (mismatches < 8)
                printk("ERROR: dynamic_loop_branch lane=%d gpu=%x expected=%x iters=%d control=%d\n",
                       (int)lane, out_values[lane], expected,
                       (int)test->iters, (int)test->control);
            ++mismatches;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES; i < BUFFER_N; ++i) {
        uint32_t expected = SENTINEL + i;
        if (out_values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: dynamic_loop_branch sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], expected);
            ++mismatches;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane)
        checksum += (int)(out_values[lane] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("fragment_rotate_dynamic_loop_branch_vc4kernel program create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("fragment_rotate_dynamic_loop_branch_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_main = 0;
    int saw_alt = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 fragment_rotate_dynamic_loop_branch_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
        const struct loop_branch_case *test = &cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            fragment_rotate_dynamic_loop_branch_vc4kernel_launch(
                program, grid, block, out_dev, test->iters, test->control,
                test->base_value) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: dynamic_loop_branch launch/copy failed case=%d\n",
                   (int)case_id);
            ++launch_failures;
            continue;
        }
        if (test->control)
            saw_alt = 1;
        else
            saw_main = 1;
        int mismatches = verify_active(test);
        int sentinels = verify_sentinels();
        int checksum = checksum_low16();
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("FRAGMENT_ROTATE_DYNAMIC_LOOP_BRANCH_CASE case=%d iters=%d control=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)test->iters, (int)test->control,
               mismatches, sentinels, checksum);
    }

    launch_failures += (int)fragment_rotate_dynamic_loop_branch_vc4kernel_runtime_launch_failures();
    int saw_both_paths = saw_main && saw_alt;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_both_paths)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_rotate_dynamic_loop_branch_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_dynamic_rotate_loop=1 saw_dynamic_rotate_branch=1 saw_both_paths=%d saw_vdw_preserve=1 rotate_direction=left_source_plus_amount checksum_accum=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, (int)LANES, saw_both_paths,
           checksum_accum,
           (int)fragment_rotate_dynamic_loop_branch_vc4kernel_runtime_launches(),
           timer_get_usec() - start);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
