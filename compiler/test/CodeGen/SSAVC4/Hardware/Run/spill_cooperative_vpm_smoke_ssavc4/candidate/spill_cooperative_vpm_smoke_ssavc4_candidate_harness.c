#include "rpi.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_MAX_N 64u
#define SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_GUARD 32u
#define SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_BUFFER_N (SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_MAX_N + SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_GUARD)
#define SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_SENTINEL 0xdeadbeefu
#define SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_WARPS_PER_BLOCK 4u
#define SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_LANE_WIDTH 16u
#define SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_CONST_SUM 406u

static uint32_t out_values[SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_BUFFER_N];

static uint32_t expected_value(uint32_t i) {
    return (2u * i) + SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_CONST_SUM;
}

static void fill_host_buffer(void) {
    for (uint32_t i = 0; i < SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_BUFFER_N; i++)
        out_values[i] = SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_SENTINEL;
}

static int verify_results(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t expected = expected_value(i);
        if (out_values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: spill_cooperative_vpm_smoke_ssavc4 i=%d gpu=%x expected=%x\n", (int)i, out_values[i], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinel_tail(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < n + SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_GUARD && i < SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_BUFFER_N; i++) {
        if (out_values[i] != SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: spill_cooperative_vpm_smoke_ssavc4 sentinel changed i=%d value=%x expected=%x\n",
                       (int)i, out_values[i], SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_values(const uint32_t *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)values[i];
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    const uint32_t warpsPerBlock = SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_WARPS_PER_BLOCK;
    const uint32_t laneWidth = SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_LANE_WIDTH;
    const uint32_t n = SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_MAX_N;

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("spill_cooperative_vpm_smoke_ssavc4 device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(warpsPerBlock * laneWidth, 1, 1);

    printk("Running VC4 spill_cooperative_vpm_smoke_ssavc4 candidate bundle...\n");
    fill_host_buffer();
    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        spill_cooperative_vpm_smoke_ssavc4_launch(program, grid, block, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
        printk("ERROR: spill_cooperative_vpm_smoke_ssavc4 launch/copy failed\n");
        launch_failures++;
    } else {
        total_mismatches += verify_results(n);
        sentinel_mismatches += verify_sentinel_tail(n);
        checksum_accum += checksum_values(out_values, n);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=spill_cooperative_vpm_smoke_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, 1, total_mismatches, sentinel_mismatches, launch_failures,
           (int)warpsPerBlock, (int)laneWidth, SPILL_COOPERATIVE_VPM_SMOKE_SSAVC4_MAX_N,
           checksum_accum, 1, 1, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
