#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_EPSILON 0.0001f
#define CHECKSUM_SCALE 1024.0f
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_MAX_N 257u
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_GUARD 32u
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_ACTIVE_QPUS 12u
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_LANE_WIDTH 16u
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_ELEMENTS_PER_WAVE (SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_ACTIVE_QPUS * SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_LANE_WIDTH)
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_MAX_WAVES ((SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_MAX_N + SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_ELEMENTS_PER_WAVE - 1u) / SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_ELEMENTS_PER_WAVE)
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_MAX_COVERAGE_N (SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_MAX_WAVES * SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_ELEMENTS_PER_WAVE)
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_BUFFER_N (SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_MAX_COVERAGE_N + SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_GUARD)
#define SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_SENTINEL (-12345.0f)

struct scalar_f32_uniform_order_canary_case {
    uint32_t n;
    float alpha;
    float beta;
};

static float out_values[SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_BUFFER_N];
static float expected_values[SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_BUFFER_N];

static const struct scalar_f32_uniform_order_canary_case test_cases[] = {
    {0u, 1.0f, -2.0f},
    {1u, -2.25f, 0.125f},
    {17u, 0.5f, 2.25f},
    {193u, 1.5f, -3.75f},
    {257u, -0.75f, 4.0f},
};

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static void fill_inputs(uint32_t n) {
    for (uint32_t i = 0; i < SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_BUFFER_N; i++) {
        out_values[i] = ((float)((i * 13u + 5u) % 89u) * 0.125f) - 3.0f;
        expected_values[i] = out_values[i];
    }
    for (uint32_t i = n; i < SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_BUFFER_N; i++) {
        out_values[i] = SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_SENTINEL;
        expected_values[i] = SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_SENTINEL;
    }
}

static void run_cpu_reference(float alpha, float beta, uint32_t n) {
    float expected = alpha * 3.0f + beta;
    for (uint32_t i = 0; i < n; i++) {
        expected_values[i] = expected;
    }
}

static int scaled_checksum(const float *values, uint32_t n) {
    int c = 0;
    for (uint32_t i = 0; i < n; i++) {
        c += (int)(values[i] * CHECKSUM_SCALE);
    }
    return c;
}

static void verify_results(uint32_t n, int *mismatch_count, float *max_abs_diff) {
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float diff = out_values[i] - expected_values[i];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff) *max_abs_diff = ad;
        if (ad > SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_EPSILON) {
            if (*mismatch_count < 8) {
                printk("ERROR: n=%d i=%d gpu=%f cpu=%f diff=%f\n", (int)n, (int)i, out_values[i], expected_values[i], diff);
            }
            (*mismatch_count)++;
        }
    }
}

static int verify_sentinel_region(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_BUFFER_N; i++) {
        if (out_values[i] != SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_SENTINEL) {
            if (mismatches < 8) {
                printk("ERROR: sentinel changed n=%d i=%d value=%f expected=%f\n", (int)n, (int)i, out_values[i], SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_SENTINEL);
            }
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("vc4_program_create failed");

    uint32_t activeQpus = SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_ACTIVE_QPUS;
    uint32_t laneWidth = SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_LANE_WIDTH;
    uint32_t elements_per_wave = activeQpus * laneWidth;
    uint32_t max_waves = (SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_MAX_N + elements_per_wave - 1u) / elements_per_wave;
    uint32_t max_coverage_n = max_waves * elements_per_wave;
    uint32_t buffer_n = max_coverage_n + SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_GUARD;
    vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

    printk("Running VC4 scalar_f32_uniform_order_canary_ssavc4 candidate bundle...\n");
    printk("VC4_RUNTIME_LAYOUT fixture=scalar_f32_uniform_order_canary_ssavc4 program_allocations=1 heap_api=1\n");

    int start = timer_get_usec();
    int totalMismatches = 0, sentinelMismatches = 0, launchFailures = 0, checksumAccum = 0;
    float maxAbsDiffOverall = 0.0f;
    const uint32_t caseCount = sizeof(test_cases) / sizeof(test_cases[0]);

    for (uint32_t caseIndex = 0; caseIndex < caseCount; caseIndex++) {
        uint32_t n = test_cases[caseIndex].n;
        float alpha = test_cases[caseIndex].alpha;
        float beta = test_cases[caseIndex].beta;
        fill_inputs(n);
        run_cpu_reference(alpha, beta, n);
        vc4_deviceptr_t out_dev = 0;
        uint32_t bytes = buffer_n * sizeof(float);
        if (vc4_m2_malloc(program, &out_dev, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0) {
            launchFailures++;
            continue;
        }
        uint32_t waves = n == 0u ? 0u : (n + elements_per_wave - 1u) / elements_per_wave;
        uint32_t roundedCoverage = waves * elements_per_wave;
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        printk("VC4_KERNEL_LAUNCH name=scalar_f32_uniform_order_canary_ssavc4 case=%u n=%u alpha=%f beta=%f waves=%u coverage=%u\n",
               caseIndex, n, alpha, beta, waves, roundedCoverage);
        if (scalar_f32_uniform_order_canary_ssavc4_launch(program, grid, block, out_dev, alpha, n, beta) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: scalar_f32_uniform_order_canary_ssavc4 launch/copy failed for n=%d\n", (int)n);
            launchFailures++;
        }
        vc4Free(program, out_dev);
        int mismatches = 0;
        float maxAbsDiff = 0.0f;
        verify_results(n, &mismatches, &maxAbsDiff);
        int caseSentinelMismatches = verify_sentinel_region(n);
        int checksum = scaled_checksum(out_values, n);
        int expectedChecksum = scaled_checksum(expected_values, n);
        if (checksum != expectedChecksum) {
            printk("ERROR: checksum mismatch n=%d gpu=%d cpu=%d\n", (int)n, checksum, expectedChecksum);
            mismatches++;
        }
        if (maxAbsDiff > maxAbsDiffOverall) maxAbsDiffOverall = maxAbsDiff;
        totalMismatches += mismatches;
        sentinelMismatches += caseSentinelMismatches;
        checksumAccum += checksum;
        printk("SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_CASE n=%d alpha=%f beta=%f qpus=%d lanes=%d coverage=%d buffer_n=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
               (int)n, alpha, beta, (int)activeQpus, (int)laneWidth, (int)roundedCoverage, (int)buffer_n,
               mismatches, caseSentinelMismatches, checksum, maxAbsDiff, (int)(caseIndex + 1), 1);
    }
    int elapsed = timer_get_usec() - start;
    const char *status = (totalMismatches == 0 && sentinelMismatches == 0 && launchFailures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=scalar_f32_uniform_order_canary_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)caseCount, totalMismatches, sentinelMismatches, launchFailures, (int)activeQpus, (int)laneWidth,
           SCALAR_F32_UNIFORM_ORDER_CANARY_SSAVC4_MAX_N, checksumAccum, maxAbsDiffOverall, 1, (int)caseCount, elapsed);
    vc4_program_destroy(program);
}
