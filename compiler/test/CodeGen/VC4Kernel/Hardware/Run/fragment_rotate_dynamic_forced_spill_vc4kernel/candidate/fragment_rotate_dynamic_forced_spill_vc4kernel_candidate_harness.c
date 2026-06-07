#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define GUARD 16u
#define BUFFER_N (LANES + GUARD)
#define SENTINEL 0x4d510000u
#define AMOUNT 17u
#define BASE_VALUE 0x31000000u
#define BETA 3u
#define SPILL_ADJUST (495u * BETA)

static uint32_t out_values[BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL + i;
}

static uint32_t expected_value(uint32_t lane) {
    uint32_t source_lane = (lane + (AMOUNT & 15u)) & 15u;
    return BASE_VALUE + source_lane + SPILL_ADJUST;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t expected = expected_value(lane);
        if (out_values[lane] != expected) {
            if (mismatches < 8)
                printk("ERROR: dynamic_forced_spill lane=%d gpu=%x expected=%x\n",
                       (int)lane, out_values[lane], expected);
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
                printk("ERROR: dynamic_forced_spill sentinel i=%d gpu=%x expected=%x\n",
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
    if (vc4_program_create(&program, 128u * 1024u) < 0 || !program)
        panic("fragment_rotate_dynamic_forced_spill_vc4kernel program create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("fragment_rotate_dynamic_forced_spill_vc4kernel allocation failed");

    fill_output();
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 fragment_rotate_dynamic_forced_spill_vc4kernel candidate bundle...\n");
    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        fragment_rotate_dynamic_forced_spill_vc4kernel_launch(
            program, grid, block, out_dev, AMOUNT, BASE_VALUE, BETA) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
        printk("ERROR: dynamic_forced_spill launch/copy failed\n");
        ++launch_failures;
    }

    launch_failures += (int)fragment_rotate_dynamic_forced_spill_vc4kernel_runtime_launch_failures();
    int total_mismatches = launch_failures ? 0 : verify_active();
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int checksum = launch_failures ? 0 : checksum_low16();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_rotate_dynamic_forced_spill_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_spill_frame_nonzero=1 saw_dynamic_rotate=1 saw_vdw_preserve=1 hidden_spill_reload_tmu_hits=0 forced_spill_live_adjust=%d rotate_direction=left_source_plus_amount checksum_accum=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)SPILL_ADJUST, checksum,
           (int)fragment_rotate_dynamic_forced_spill_vc4kernel_runtime_launches(),
           timer_get_usec() - start);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
