#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_MAX_N 257u
#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_GUARD 32u
#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_SENTINEL 0xdeadbeefu
#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_ACTIVE_QPUS 12u
#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_LANE_WIDTH 16u
#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_ELEMENTS_PER_WAVE (RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_ACTIVE_QPUS * RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_LANE_WIDTH)
#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_MAX_WAVES ((RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_MAX_N + RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_ELEMENTS_PER_WAVE - 1u) / RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_ELEMENTS_PER_WAVE)
#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_MAX_COVERAGE_N (RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_MAX_WAVES * RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_ELEMENTS_PER_WAVE)
#define RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_BUFFER_N (RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_MAX_COVERAGE_N + RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_GUARD)

static const uint32_t test_sizes[] = {193u, 257u};

static uint32_t out_values[RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_BUFFER_N];

static uint32_t expected_value(uint32_t case_id, uint32_t i) {
    return 0x51000000u | ((case_id & 0xffu) << 16) | (i & 0xffffu);
}

static void fill_host_buffer(void) {
    for (uint32_t i = 0; i < RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_BUFFER_N; i++)
        out_values[i] = RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_SENTINEL;
}

static int verify_results(uint32_t case_id, uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t expected = expected_value(case_id, i);
        if (out_values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: resource_limited_vpm_residency_ssavc4 i=%d gpu=%x expected=%x\n", (int)i, out_values[i], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinel_region(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_BUFFER_N; i++) {
        if (out_values[i] != RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: resource_limited_vpm_residency_ssavc4 sentinel changed i=%d value=%x expected=%x\n", (int)i, out_values[i], RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(const uint32_t *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    const uint32_t activeQpus = RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_ACTIVE_QPUS;
    const uint32_t laneWidth = RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_LANE_WIDTH;
    const uint32_t elementsPerWave = activeQpus * laneWidth;
    const uint32_t caseCount = sizeof(test_sizes) / sizeof(test_sizes[0]);

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("resource_limited_vpm_residency_ssavc4 device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

    printk("Running VC4 resource_limited_vpm_residency_ssavc4 candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < caseCount; case_id++) {
        uint32_t n = test_sizes[case_id];
        uint32_t waves = (n + elementsPerWave - 1u) / elementsPerWave;
        uint32_t roundedCoverage = waves * elementsPerWave;
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);

        fill_host_buffer();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            resource_limited_vpm_residency_ssavc4_launch(program, grid, block, out_dev, n, case_id) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: resource_limited_vpm_residency_ssavc4 launch/copy failed n=%d waves=%d\n", (int)n, (int)waves);
            launch_failures++;
            continue;
        }

        int mismatches = verify_results(case_id, n);
        int caseSentinelMismatches = verify_sentinel_region(n);
        int checksum = checksum_low16(out_values, n);
        total_mismatches += mismatches;
        sentinel_mismatches += caseSentinelMismatches;
        checksum_accum += checksum;

        printk("RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_CASE case=%d n=%d waves=%d requests=%d rounded_coverage=%d buffer_n=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)n, (int)waves, (int)(waves * activeQpus),
               (int)roundedCoverage, RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_BUFFER_N,
               mismatches, caseSentinelMismatches, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=resource_limited_vpm_residency_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)caseCount, total_mismatches, sentinel_mismatches,
           launch_failures, (int)activeQpus, (int)laneWidth,
           RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_MAX_N,
           RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_MAX_COVERAGE_N,
           RESOURCE_LIMITED_VPM_RESIDENCY_SSAVC4_BUFFER_N,
           checksum_accum, 1, (int)caseCount, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
