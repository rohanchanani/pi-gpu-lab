#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SAXPY_FULL_EPSILON 0.0001f
#define CHECKSUM_SCALE 1024.0f
#define SAXPY_FULL_MAX_N 1000u
#define SAXPY_FULL_GUARD 32u
#define SAXPY_FULL_BUFFER_N (SAXPY_FULL_MAX_N + SAXPY_FULL_GUARD)
#define SAXPY_FULL_SENTINEL (-12345.0f)

static float x_values[SAXPY_FULL_BUFFER_N];
static float y_values[SAXPY_FULL_BUFFER_N];
static float y_initial[SAXPY_FULL_BUFFER_N];
static float expected_values[SAXPY_FULL_BUFFER_N];

static const uint32_t test_sizes[] = {0u,1u,2u,15u,16u,17u,31u,32u,33u,191u,192u,193u,255u,256u,257u,767u,768u,769u,1000u};

static float absf_local(float value) { return value < 0.0f ? -value : value; }
static void fill_inputs(uint32_t n) {
    for (uint32_t i = 0; i < SAXPY_FULL_BUFFER_N; i++) {
        x_values[i] = ((float)((i * 7u + 3u) % 101u) * 0.125f) - 4.0f;
        y_values[i] = ((float)((i * 5u + 11u) % 67u) * 0.25f) + 0.5f;
        y_initial[i] = y_values[i];
        expected_values[i] = y_values[i];
    }
    for (uint32_t i = n; i < n + SAXPY_FULL_GUARD && i < SAXPY_FULL_BUFFER_N; i++) {
        y_values[i] = SAXPY_FULL_SENTINEL;
        y_initial[i] = SAXPY_FULL_SENTINEL;
        expected_values[i] = SAXPY_FULL_SENTINEL;
    }
}
static void run_cpu_reference(float alpha, uint32_t n) { for (uint32_t i = 0; i < n; i++) expected_values[i] = alpha * x_values[i] + y_initial[i]; }
static int scaled_checksum(const float *values, uint32_t n) { int c = 0; for (uint32_t i = 0; i < n; i++) c += (int)(values[i] * CHECKSUM_SCALE); return c; }
static void verify_results(uint32_t n, int *mismatch_count, float *max_abs_diff) {
    *mismatch_count = 0; *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float diff = y_values[i] - expected_values[i];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff) *max_abs_diff = ad;
        if (ad > SAXPY_FULL_EPSILON) {
            if (*mismatch_count < 8) printk("ERROR: n=%d i=%d gpu=%f cpu=%f diff=%f\n", (int)n, (int)i, y_values[i], expected_values[i], diff);
            (*mismatch_count)++;
        }
    }
}
static int verify_sentinel_tail(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < n + SAXPY_FULL_GUARD && i < SAXPY_FULL_BUFFER_N; i++) {
        if (y_values[i] != SAXPY_FULL_SENTINEL) {
            if (mismatches < 8) printk("ERROR: sentinel changed n=%d i=%d value=%f expected=%f\n", (int)n, (int)i, y_values[i], SAXPY_FULL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    const float alpha = 2.5f;
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("vc4_program_create failed");

    uint32_t activeQpus = 12u;
    uint32_t laneWidth = 16u;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

    printk("Running VC4 saxpy_full M2 candidate bundle...\n");
    printk("VC4_RUNTIME_LAYOUT fixture=saxpy_full program_allocations=1 heap_api=1\n");

    int start = timer_get_usec();
    int totalMismatches = 0, sentinelMismatches = 0, launchFailures = 0, checksumAccum = 0;
    uint32_t checkedElements = 0;
    float maxAbsDiffOverall = 0.0f;
    const uint32_t caseCount = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (uint32_t caseIndex = 0; caseIndex < caseCount; caseIndex++) {
        uint32_t n = test_sizes[caseIndex];
        fill_inputs(n);
        run_cpu_reference(alpha, n);
        vc4_deviceptr_t x_dev = 0, y_dev = 0;
        uint32_t bytes = SAXPY_FULL_BUFFER_N * sizeof(float);
        if (vc4_m2_malloc(program, &x_dev, bytes) < 0 || vc4_m2_malloc(program, &y_dev, bytes) < 0 ||
            vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 || vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0) {
            launchFailures++;
            continue;
        }
        printk("VC4_KERNEL_LAUNCH name=saxpy_full case=%u n=%u\n", caseIndex, n);
        if (saxpy_full_launch(program, grid, block, x_dev, y_dev, alpha, n) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, bytes) < 0) {
            printk("ERROR: saxpy_full launch/copy failed for n=%d\n", (int)n);
            launchFailures++;
        }
        vc4Free(program, x_dev);
        vc4Free(program, y_dev);
        int mismatches = 0; float maxAbsDiff = 0.0f;
        verify_results(n, &mismatches, &maxAbsDiff);
        int caseSentinelMismatches = verify_sentinel_tail(n);
        int checksum = scaled_checksum(y_values, n);
        int expectedChecksum = scaled_checksum(expected_values, n);
        if (checksum != expectedChecksum) { printk("ERROR: checksum mismatch n=%d gpu=%d cpu=%d\n", (int)n, checksum, expectedChecksum); mismatches++; }
        if (maxAbsDiff > maxAbsDiffOverall) maxAbsDiffOverall = maxAbsDiff;
        totalMismatches += mismatches;
        sentinelMismatches += caseSentinelMismatches;
        checksumAccum += checksum;
        checkedElements += n;
        printk("SAXPY_FULL_CASE n=%d qpus=%d lanes=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n", (int)n, (int)activeQpus, (int)laneWidth, mismatches, caseSentinelMismatches, checksum, maxAbsDiff, (int)(caseIndex + 1), 1);
    }
    int elapsed = timer_get_usec() - start;
    uint32_t runtimeAllocations = saxpy_full_runtime_allocations();
    uint32_t runtimeLaunches = saxpy_full_runtime_launches();
    uint32_t runtimeCapacity = saxpy_full_runtime_capacity();
    uint32_t codeUploads = saxpy_full_runtime_code_uploads();
    uint32_t recordedLaunchFailures = saxpy_full_runtime_launch_failures();
    const char *status = (totalMismatches == 0 && sentinelMismatches == 0 && launchFailures == 0) ? "PASS" : "FAIL";
    if (runtimeAllocations != 1u || runtimeLaunches != caseCount || codeUploads != 1u ||
        recordedLaunchFailures != 0u)
        status = "FAIL";
    printk("VC4_TEST_RESULT name=saxpy_full status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_capacity=%d code_uploads=%d elapsed_usec=%d\n", status, (int)caseCount, (int)checkedElements, totalMismatches, sentinelMismatches, launchFailures, (int)recordedLaunchFailures, (int)activeQpus, (int)laneWidth, SAXPY_FULL_MAX_N, checksumAccum, maxAbsDiffOverall, (int)runtimeAllocations, (int)runtimeLaunches, (int)runtimeCapacity, (int)codeUploads, elapsed);
    vc4_program_destroy(program);
}
