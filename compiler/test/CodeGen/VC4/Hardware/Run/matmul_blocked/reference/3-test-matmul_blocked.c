#include "rpi.h"
#include "matmul_blocked_launch.h"

#define MATMUL_BLOCKED_EPSILON 0.001f
#define CHECKSUM_SCALE 1024.0f
#define MATMUL_BLOCKED_MAX_M 25u
#define MATMUL_BLOCKED_MAX_N 31u
#define MATMUL_BLOCKED_MAX_K 25u
#define MATMUL_BLOCKED_GUARD 64u
#define MATMUL_BLOCKED_MAX_A (MATMUL_BLOCKED_MAX_M * MATMUL_BLOCKED_MAX_K)
#define MATMUL_BLOCKED_MAX_B (MATMUL_BLOCKED_MAX_K * MATMUL_BLOCKED_MAX_N)
#define MATMUL_BLOCKED_MAX_C (MATMUL_BLOCKED_MAX_M * MATMUL_BLOCKED_MAX_N)
#define MATMUL_BLOCKED_C_BUFFER (MATMUL_BLOCKED_MAX_C + MATMUL_BLOCKED_GUARD)
#define MATMUL_BLOCKED_SENTINEL (-9876.0f)
#define MATMUL_BLOCKED_EXPECTED_TILE_WAVES 14u

struct matmul_blocked_case
{
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

static const struct matmul_blocked_case cases[] = {
    {0u, 7u, 5u},
    {4u, 0u, 5u},
    {5u, 6u, 0u},
    {1u, 1u, 1u},
    {4u, 7u, 5u},
    {12u, 16u, 12u},
    {13u, 17u, 7u},
    {25u, 31u, 25u},
};

static float a_values[MATMUL_BLOCKED_MAX_A];
static float b_values[MATMUL_BLOCKED_MAX_B];
static float c_values[MATMUL_BLOCKED_C_BUFFER];
static float expected_values[MATMUL_BLOCKED_C_BUFFER];

static float absf_local(float value)
{
    return value < 0.0f ? -value : value;
}

static void fill_inputs(uint32_t case_index, uint32_t m, uint32_t n, uint32_t k)
{
    for (uint32_t i = 0; i < MATMUL_BLOCKED_MAX_A; i++)
        a_values[i] = 0.0f;
    for (uint32_t i = 0; i < MATMUL_BLOCKED_MAX_B; i++)
        b_values[i] = 0.0f;
    for (uint32_t i = 0; i < MATMUL_BLOCKED_C_BUFFER; i++)
    {
        c_values[i] = MATMUL_BLOCKED_SENTINEL;
        expected_values[i] = MATMUL_BLOCKED_SENTINEL;
    }

    for (uint32_t row = 0; row < m; row++)
    {
        for (uint32_t col = 0; col < k; col++)
        {
            int raw = (int)((row * 7u + col * 5u + case_index * 3u) % 23u) - 11;
            a_values[row * k + col] = (float)raw * 0.0625f;
        }
    }

    for (uint32_t row = 0; row < k; row++)
    {
        for (uint32_t col = 0; col < n; col++)
        {
            int raw = (int)((row * 11u + col * 3u + case_index * 5u) % 29u) - 14;
            b_values[row * n + col] = (float)raw * 0.03125f;
        }
    }
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

static void verify_results(
    uint32_t case_index,
    uint32_t m,
    uint32_t n,
    int *mismatch_count,
    float *max_abs_diff)
{
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;

    uint32_t count = m * n;
    for (uint32_t i = 0; i < count; i++)
    {
        float diff = c_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;

        if (abs_diff > MATMUL_BLOCKED_EPSILON)
        {
            if (*mismatch_count < 8)
            {
                uint32_t row = n ? (i / n) : 0;
                uint32_t col = n ? (i % n) : 0;
                printk("ERROR: case=%d row=%d col=%d gpu=%f cpu=%f diff=%f\n",
                       (int)case_index,
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

static int verify_sentinel_tail(uint32_t m, uint32_t n)
{
    uint32_t count = m * n;
    int mismatches = 0;

    for (uint32_t i = count; i < count + MATMUL_BLOCKED_GUARD && i < MATMUL_BLOCKED_C_BUFFER; i++)
    {
        if (c_values[i] != MATMUL_BLOCKED_SENTINEL)
        {
            if (mismatches < 8)
            {
                printk("ERROR: sentinel changed index=%d value=%f expected=%f\n",
                       (int)i,
                       c_values[i],
                       MATMUL_BLOCKED_SENTINEL);
            }
            mismatches++;
        }
    }

    return mismatches;
}

static uint32_t ceil_div_u32(uint32_t a, uint32_t b)
{
    if (a == 0)
        return 0;
    return (a + b - 1u) / b;
}

static uint32_t expected_tile_waves(uint32_t m, uint32_t n)
{
    return ceil_div_u32(m, 12u) * ceil_div_u32(n, 16u);
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
    if (laneWidth != VC4_RUNTIME_LANE_WIDTH)
        panic("Unexpected lane width: %d", (int)laneWidth);

    printk("Running VC4 matmul_blocked reference bundle...\n");

    if (matmul_blocked_prepare(&rt,
                               MATMUL_BLOCKED_MAX_M,
                               MATMUL_BLOCKED_MAX_N,
                               MATMUL_BLOCKED_MAX_K) < 0)
        panic("matmul_blocked runtime setup failed");

    printk("MATMUL_BLOCKED_RUNTIME_SETUP max_m=%d max_n=%d max_k=%d allocations=%d ident1=%x vpmbase=%x\n",
           MATMUL_BLOCKED_MAX_M,
           MATMUL_BLOCKED_MAX_N,
           MATMUL_BLOCKED_MAX_K,
           (int)matmul_blocked_runtime_allocations(),
           (unsigned)matmul_blocked_runtime_ident1(),
           (unsigned)matmul_blocked_runtime_vpmbase_readback());

    int start = timer_get_usec();
    int totalMismatches = 0;
    int sentinelMismatches = 0;
    int launchFailures = 0;
    int checksumAccum = 0;
    float maxAbsDiffOverall = 0.0f;
    uint32_t expectedWaves = 0;

    const uint32_t caseCount = sizeof(cases) / sizeof(cases[0]);

    for (uint32_t caseIndex = 0; caseIndex < caseCount; caseIndex++)
    {
        uint32_t m = cases[caseIndex].m;
        uint32_t n = cases[caseIndex].n;
        uint32_t k = cases[caseIndex].k;

        if (m > MATMUL_BLOCKED_MAX_M || n > MATMUL_BLOCKED_MAX_N || k > MATMUL_BLOCKED_MAX_K)
            panic("matmul_blocked case exceeds max shape");

        expectedWaves += expected_tile_waves(m, n);

        fill_inputs(caseIndex, m, n, k);
        run_cpu_reference(m, n, k);

        if (matmul_blocked_launch(&rt, a_values, b_values, c_values, m, n, k) < 0)
        {
            printk("ERROR: matmul_blocked launch failed case=%d m=%d n=%d k=%d\n",
                   (int)caseIndex,
                   (int)m,
                   (int)n,
                   (int)k);
            launchFailures++;
            break;
        }

        int mismatches = 0;
        float maxAbsDiff = 0.0f;
        verify_results(caseIndex, m, n, &mismatches, &maxAbsDiff);

        int caseSentinelMismatches = verify_sentinel_tail(m, n);
        uint32_t count = m * n;
        int checksum = scaled_checksum(c_values, count);
        int expectedChecksum = scaled_checksum(expected_values, count);
        if (checksum != expectedChecksum)
        {
            printk("ERROR: checksum mismatch case=%d gpu=%d cpu=%d\n",
                   (int)caseIndex,
                   checksum,
                   expectedChecksum);
            mismatches++;
        }

        if (maxAbsDiff > maxAbsDiffOverall)
            maxAbsDiffOverall = maxAbsDiff;

        totalMismatches += mismatches;
        sentinelMismatches += caseSentinelMismatches;
        checksumAccum += checksum;

        printk("MATMUL_BLOCKED_CASE case=%d m=%d n=%d k=%d qpus=%d lanes=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d tile_waves=%d allocations=%d\n",
               (int)caseIndex,
               (int)m,
               (int)n,
               (int)k,
               (int)activeQpus,
               (int)laneWidth,
               mismatches,
               caseSentinelMismatches,
               checksum,
               maxAbsDiff,
               (int)matmul_blocked_runtime_launches(),
               (int)matmul_blocked_runtime_tile_waves(),
               (int)matmul_blocked_runtime_allocations());
    }

    int end = timer_get_usec();
    int elapsed = end - start;

    uint32_t runtimeAllocations = matmul_blocked_runtime_allocations();
    uint32_t runtimeLaunches = matmul_blocked_runtime_launches();
    uint32_t runtimeTileWaves = matmul_blocked_runtime_tile_waves();
    uint32_t runtimeTimeouts = matmul_blocked_runtime_timeouts();
    uint32_t errstatRelevantChanged = matmul_blocked_runtime_errstat_relevant_changed();

    if (expectedWaves != MATMUL_BLOCKED_EXPECTED_TILE_WAVES)
        printk("ERROR: expected tile wave accounting got=%d expected=%d\n",
               (int)expectedWaves,
               MATMUL_BLOCKED_EXPECTED_TILE_WAVES);

    const char *status =
        (totalMismatches == 0 &&
         sentinelMismatches == 0 &&
         launchFailures == 0 &&
         runtimeAllocations == 1u &&
         runtimeLaunches == caseCount &&
         expectedWaves == MATMUL_BLOCKED_EXPECTED_TILE_WAVES &&
         runtimeTileWaves == expectedWaves &&
         runtimeTimeouts == 0u &&
         errstatRelevantChanged == 0u) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=matmul_blocked status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_m=%d max_n=%d max_k=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_tile_waves=%d timeouts=%d errstat_relevant_changed=%d srqcs_after_last_wave=%x elapsed_usec=%d\n",
           status,
           (int)caseCount,
           totalMismatches,
           sentinelMismatches,
           launchFailures,
           (int)activeQpus,
           (int)laneWidth,
           MATMUL_BLOCKED_MAX_M,
           MATMUL_BLOCKED_MAX_N,
           MATMUL_BLOCKED_MAX_K,
           checksumAccum,
           maxAbsDiffOverall,
           (int)runtimeAllocations,
           (int)runtimeLaunches,
           (int)runtimeTileWaves,
           (int)runtimeTimeouts,
           (int)errstatRelevantChanged,
           (unsigned)matmul_blocked_runtime_srqcs_after_last_wave(),
           elapsed);

    vc4_runtime_shutdown(&rt);
}
