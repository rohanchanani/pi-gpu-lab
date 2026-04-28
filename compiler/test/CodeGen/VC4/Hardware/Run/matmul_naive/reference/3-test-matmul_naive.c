#include "rpi.h"
#include "matmul_naive_launch.h"

#define MATMUL_NAIVE_EPSILON 0.0005f
#define CHECKSUM_SCALE 4096.0f

#define MATMUL_NAIVE_CASES 10
#define MATMUL_NAIVE_MAX_M 19u
#define MATMUL_NAIVE_MAX_N 31u
#define MATMUL_NAIVE_MAX_K 13u
#define MATMUL_NAIVE_MAX_A (MATMUL_NAIVE_MAX_M * MATMUL_NAIVE_MAX_K)
#define MATMUL_NAIVE_MAX_B (MATMUL_NAIVE_MAX_K * MATMUL_NAIVE_MAX_N)
#define MATMUL_NAIVE_MAX_C (MATMUL_NAIVE_MAX_M * MATMUL_NAIVE_MAX_N)
#define MATMUL_NAIVE_HOST_GUARD 16u
#define MATMUL_NAIVE_SENTINEL -7777.0f

typedef struct
{
    uint32_t m;
    uint32_t n;
    uint32_t k;
} matmul_case_t;

static const matmul_case_t cases[MATMUL_NAIVE_CASES] = {
    {0, 7, 3},
    {4, 0, 5},
    {4, 7, 0},
    {1, 1, 1},
    {3, 5, 4},
    {5, 15, 6},
    {5, 16, 6},
    {5, 17, 6},
    {13, 17, 9},
    {19, 31, 13},
};

static float a_values[MATMUL_NAIVE_MAX_A ? MATMUL_NAIVE_MAX_A : 1];
static float b_values[MATMUL_NAIVE_MAX_B ? MATMUL_NAIVE_MAX_B : 1];
static float c_values[MATMUL_NAIVE_MAX_C + MATMUL_NAIVE_HOST_GUARD];
static float expected_values[MATMUL_NAIVE_MAX_C ? MATMUL_NAIVE_MAX_C : 1];

static float absf_local(float value)
{
    return value < 0.0f ? -value : value;
}

static float make_a_value(uint32_t i)
{
    int centered = (int)(i % 17u) - 8;
    return ((float)centered) * 0.125f;
}

static float make_b_value(uint32_t i)
{
    int centered = (int)(i % 19u) - 9;
    return ((float)centered) * 0.0625f;
}

static void fill_inputs(uint32_t m, uint32_t n, uint32_t k)
{
    uint32_t aCount = m * k;
    uint32_t bCount = k * n;
    uint32_t cCount = m * n;

    for (uint32_t i = 0; i < aCount; i++)
        a_values[i] = make_a_value(i + 3u * m + 5u * k);
    for (uint32_t i = 0; i < bCount; i++)
        b_values[i] = make_b_value(i + 7u * n + 11u * k);
    for (uint32_t i = 0; i < cCount + MATMUL_NAIVE_HOST_GUARD; i++)
        c_values[i] = MATMUL_NAIVE_SENTINEL;
}

static void run_cpu_reference(uint32_t m, uint32_t n, uint32_t k)
{
    for (uint32_t row = 0; row < m; row++)
    {
        for (uint32_t col = 0; col < n; col++)
        {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < k; kk++)
                acc += a_values[row * k + kk] * b_values[kk * n + col];
            expected_values[row * n + col] = acc;
        }
    }
}

static int scaled_checksum(const float *values, uint32_t count)
{
    int checksum = 0;
    for (uint32_t i = 0; i < count; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t m,
                           uint32_t n,
                           int *mismatch_count,
                           float *max_abs_diff)
{
    uint32_t cCount = m * n;
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;

    for (uint32_t i = 0; i < cCount; i++)
    {
        float diff = c_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;

        if (abs_diff > MATMUL_NAIVE_EPSILON)
        {
            if (*mismatch_count < 8)
            {
                uint32_t row = n ? (i / n) : 0;
                uint32_t col = n ? (i % n) : 0;
                printk("ERROR: row=%d col=%d gpu=%f cpu=%f diff=%f\n",
                       (int)row,
                       (int)col,
                       c_values[i],
                       expected_values[i],
                       diff);
            }
            (*mismatch_count)++;
        }
    }
}

static int verify_host_guard(uint32_t m, uint32_t n)
{
    uint32_t cCount = m * n;
    int mismatches = 0;
    for (uint32_t i = 0; i < MATMUL_NAIVE_HOST_GUARD; i++)
    {
        if (c_values[cCount + i] != MATMUL_NAIVE_SENTINEL)
        {
            if (mismatches < 4)
            {
                printk("ERROR: host guard i=%d value=%f\n",
                       (int)i,
                       c_values[cCount + i]);
            }
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void)
{
    struct vc4_runtime rt;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    uint32_t activeQpus = vc4_runtime_active_qpus(&rt);
    uint32_t laneWidth = vc4_runtime_lane_width();

    if (activeQpus != VC4_RUNTIME_MAX_QPUS)
        panic("Unexpected active QPU count: %d", (int)activeQpus);
    if (laneWidth != 16)
        panic("Unexpected lane width: %d", (int)laneWidth);

    if (matmul_naive_prepare(&rt,
                             MATMUL_NAIVE_MAX_M,
                             MATMUL_NAIVE_MAX_N,
                             MATMUL_NAIVE_MAX_K) < 0)
        panic("matmul_naive one-time runtime setup failed");

    printk("Running VC4 matmul_naive reference bundle...\n");
    printk("MATMUL_NAIVE_RUNTIME_SETUP max_m=%d max_n=%d max_k=%d allocations=%d\n",
           (int)MATMUL_NAIVE_MAX_M,
           (int)MATMUL_NAIVE_MAX_N,
           (int)MATMUL_NAIVE_MAX_K,
           (int)matmul_naive_runtime_allocations());

    int totalMismatches = 0;
    int sentinelMismatches = 0;
    int launchFailures = 0;
    int checksumAccum = 0;
    float globalMaxAbsDiff = 0.0f;

    int start = timer_get_usec();

    for (uint32_t caseIndex = 0; caseIndex < MATMUL_NAIVE_CASES; caseIndex++)
    {
        uint32_t m = cases[caseIndex].m;
        uint32_t n = cases[caseIndex].n;
        uint32_t k = cases[caseIndex].k;
        uint32_t cCount = m * n;

        fill_inputs(m, n, k);
        run_cpu_reference(m, n, k);

        if (matmul_naive_launch(&rt, a_values, b_values, c_values, m, n, k) < 0)
        {
            printk("MATMUL_NAIVE_CASE case=%d m=%d n=%d k=%d launch=FAIL launches=%d allocations=%d\n",
                   (int)caseIndex,
                   (int)m,
                   (int)n,
                   (int)k,
                   (int)matmul_naive_runtime_launches(),
                   (int)matmul_naive_runtime_allocations());
            launchFailures++;
            continue;
        }

        int mismatches = 0;
        float maxAbsDiff = 0.0f;
        verify_results(m, n, &mismatches, &maxAbsDiff);
        int guardMismatches = verify_host_guard(m, n);
        int checksum = scaled_checksum(c_values, cCount);
        int expectedChecksum = scaled_checksum(expected_values, cCount);

        if (checksum != expectedChecksum)
        {
            printk("ERROR: checksum case=%d gpu=%d cpu=%d\n",
                   (int)caseIndex,
                   checksum,
                   expectedChecksum);
            mismatches++;
        }

        totalMismatches += mismatches;
        sentinelMismatches += guardMismatches;
        checksumAccum += checksum;
        if (maxAbsDiff > globalMaxAbsDiff)
            globalMaxAbsDiff = maxAbsDiff;

        printk("MATMUL_NAIVE_CASE case=%d m=%d n=%d k=%d elements=%d mismatches=%d guard_mismatches=%d checksum=%d expected_checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
               (int)caseIndex,
               (int)m,
               (int)n,
               (int)k,
               (int)cCount,
               mismatches,
               guardMismatches,
               checksum,
               expectedChecksum,
               maxAbsDiff,
               (int)matmul_naive_runtime_launches(),
               (int)matmul_naive_runtime_allocations());
    }

    int end = timer_get_usec();
    int elapsed = end - start;

    uint32_t runtimeAllocations = matmul_naive_runtime_allocations();
    uint32_t runtimeLaunches = matmul_naive_runtime_launches();
    uint32_t runtimeCapacityM = matmul_naive_runtime_capacity_m();
    uint32_t runtimeCapacityN = matmul_naive_runtime_capacity_n();
    uint32_t runtimeCapacityK = matmul_naive_runtime_capacity_k();

    const char *status =
        (totalMismatches == 0 &&
         sentinelMismatches == 0 &&
         launchFailures == 0 &&
         runtimeAllocations == 1 &&
         runtimeLaunches == MATMUL_NAIVE_CASES &&
         runtimeCapacityM == MATMUL_NAIVE_MAX_M &&
         runtimeCapacityN == MATMUL_NAIVE_MAX_N &&
         runtimeCapacityK == MATMUL_NAIVE_MAX_K) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=matmul_naive status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_m=%d max_n=%d max_k=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status,
           MATMUL_NAIVE_CASES,
           totalMismatches,
           sentinelMismatches,
           launchFailures,
           (int)activeQpus,
           (int)laneWidth,
           (int)MATMUL_NAIVE_MAX_M,
           (int)MATMUL_NAIVE_MAX_N,
           (int)MATMUL_NAIVE_MAX_K,
           checksumAccum,
           globalMaxAbsDiff,
           (int)runtimeAllocations,
           (int)runtimeLaunches,
           elapsed);

    if (status[0] != 'P')
        panic("matmul_naive verification failed");

    /* Intentionally keep the single GPU allocation live until reboot.  The
     * hardware runner power-cycles before each hardware test, and this avoids
     * the repeated alloc/lock/unlock/free cycle that this test is specifically
     * avoiding. */
    vc4_runtime_shutdown(&rt);
}
