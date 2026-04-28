#include "rpi.h"
#include "saxpy_full_launch.h"

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

static const uint32_t test_sizes[] = {
    0u,
    1u,
    2u,
    15u,
    16u,
    17u,
    31u,
    32u,
    33u,
    191u,
    192u,
    193u,
    255u,
    256u,
    257u,
    767u,
    768u,
    769u,
    1000u,
};

static float absf_local(float value)
{
    return value < 0.0f ? -value : value;
}

static void fill_inputs(uint32_t n)
{
    for (uint32_t i = 0; i < SAXPY_FULL_BUFFER_N; i++)
    {
        x_values[i] = ((float)((i * 7u + 3u) % 101u) * 0.125f) - 4.0f;
        y_values[i] = ((float)((i * 5u + 11u) % 67u) * 0.25f) + 0.5f;
        y_initial[i] = y_values[i];
        expected_values[i] = y_values[i];
    }

    for (uint32_t i = n; i < n + SAXPY_FULL_GUARD && i < SAXPY_FULL_BUFFER_N; i++)
    {
        y_values[i] = SAXPY_FULL_SENTINEL;
        y_initial[i] = SAXPY_FULL_SENTINEL;
        expected_values[i] = SAXPY_FULL_SENTINEL;
    }
}

static void run_cpu_reference(float alpha, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
        expected_values[i] = alpha * x_values[i] + y_initial[i];
}

static int scaled_checksum(const float *values, uint32_t n)
{
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t n, int *mismatch_count, float *max_abs_diff)
{
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;

    for (uint32_t i = 0; i < n; i++)
    {
        float diff = y_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;

        if (abs_diff > SAXPY_FULL_EPSILON)
        {
            if (*mismatch_count < 8)
            {
                printk("ERROR: n=%d i=%d gpu=%f cpu=%f diff=%f\n",
                       (int)n,
                       (int)i,
                       y_values[i],
                       expected_values[i],
                       diff);
            }
            (*mismatch_count)++;
        }
    }
}

static int verify_sentinel_tail(uint32_t n)
{
    int mismatches = 0;
    for (uint32_t i = n; i < n + SAXPY_FULL_GUARD && i < SAXPY_FULL_BUFFER_N; i++)
    {
        if (y_values[i] != SAXPY_FULL_SENTINEL)
        {
            if (mismatches < 8)
            {
                printk("ERROR: sentinel changed n=%d i=%d value=%f expected=%f\n",
                       (int)n,
                       (int)i,
                       y_values[i],
                       SAXPY_FULL_SENTINEL);
            }
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void)
{
    struct vc4_runtime rt;
    const float alpha = 2.5f;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    uint32_t activeQpus = vc4_runtime_active_qpus(&rt);
    uint32_t laneWidth = vc4_runtime_lane_width();

    if (activeQpus != VC4_RUNTIME_MAX_QPUS)
        panic("Unexpected active QPU count: %d", (int)activeQpus);
    if (laneWidth != VC4_RUNTIME_LANE_WIDTH)
        panic("Unexpected lane width: %d", (int)laneWidth);

    if (saxpy_full_prepare(&rt, SAXPY_FULL_MAX_N) < 0)
        panic("saxpy_full one-time runtime setup failed");

    printk("Running VC4 saxpy_full reference bundle...\n");
    printk("SAXPY_FULL_RUNTIME_SETUP max_n=%d allocations=%d\n",
           SAXPY_FULL_MAX_N,
           (int)saxpy_full_runtime_allocations());

    int start = timer_get_usec();
    int totalMismatches = 0;
    int sentinelMismatches = 0;
    int launchFailures = 0;
    int checksumAccum = 0;
    float maxAbsDiffOverall = 0.0f;

    const uint32_t caseCount = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (uint32_t caseIndex = 0; caseIndex < caseCount; caseIndex++)
    {
        uint32_t n = test_sizes[caseIndex];
        if (n > SAXPY_FULL_MAX_N)
            panic("test n exceeds SAXPY_FULL_MAX_N: %d", (int)n);

        fill_inputs(n);
        run_cpu_reference(alpha, n);

        if (saxpy_full_launch(&rt, x_values, y_values, alpha, n) < 0)
        {
            printk("ERROR: saxpy_full launch failed for n=%d\n", (int)n);
            launchFailures++;
            continue;
        }

        int mismatches = 0;
        float maxAbsDiff = 0.0f;
        verify_results(n, &mismatches, &maxAbsDiff);

        int caseSentinelMismatches = verify_sentinel_tail(n);
        int checksum = scaled_checksum(y_values, n);
        int expectedChecksum = scaled_checksum(expected_values, n);
        if (checksum != expectedChecksum)
        {
            printk("ERROR: checksum mismatch n=%d gpu=%d cpu=%d\n",
                   (int)n,
                   checksum,
                   expectedChecksum);
            mismatches++;
        }

        if (maxAbsDiff > maxAbsDiffOverall)
            maxAbsDiffOverall = maxAbsDiff;

        totalMismatches += mismatches;
        sentinelMismatches += caseSentinelMismatches;
        checksumAccum += checksum;

        printk("SAXPY_FULL_CASE n=%d qpus=%d lanes=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
               (int)n,
               (int)activeQpus,
               (int)laneWidth,
               mismatches,
               caseSentinelMismatches,
               checksum,
               maxAbsDiff,
               (int)saxpy_full_runtime_launches(),
               (int)saxpy_full_runtime_allocations());
    }

    int end = timer_get_usec();
    int elapsed = end - start;

    uint32_t runtimeAllocations = saxpy_full_runtime_allocations();
    uint32_t runtimeLaunches = saxpy_full_runtime_launches();
    uint32_t runtimeCapacity = saxpy_full_runtime_capacity();

    const char *status =
        (totalMismatches == 0 &&
         sentinelMismatches == 0 &&
         launchFailures == 0 &&
         runtimeAllocations == 1 &&
         runtimeLaunches == caseCount &&
         runtimeCapacity == SAXPY_FULL_MAX_N) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=saxpy_full status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status,
           (int)caseCount,
           totalMismatches,
           sentinelMismatches,
           launchFailures,
           (int)activeQpus,
           (int)laneWidth,
           SAXPY_FULL_MAX_N,
           checksumAccum,
           maxAbsDiffOverall,
           (int)runtimeAllocations,
           (int)runtimeLaunches,
           elapsed);

    /* Intentionally keep the single GPU allocation live until reboot.  The
     * hardware runner power-cycles before each hardware test, and this avoids
     * the repeated alloc/lock/unlock/free cycle that this test is specifically
     * avoiding. */
    vc4_runtime_shutdown(&rt);
}
