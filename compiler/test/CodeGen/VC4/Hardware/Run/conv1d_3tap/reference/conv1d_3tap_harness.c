#include "rpi.h"
#include "conv1d_3tap_launch.h"

#define CONV1D_3TAP_CASES 13u
#define CONV1D_3TAP_EPSILON 0.0003f
#define CHECKSUM_SCALE 4096.0f
#define HOST_GUARD_FLOATS 32u
#define HOST_SENTINEL (-24680.0f)

static const uint32_t test_sizes[CONV1D_3TAP_CASES] = {
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
511u,
};

static float x_values[CONV1D_3TAP_MAX_N ? CONV1D_3TAP_MAX_N : 1];
static float out_values[CONV1D_3TAP_MAX_N + HOST_GUARD_FLOATS];
static float expected_values[CONV1D_3TAP_MAX_N ? CONV1D_3TAP_MAX_N : 1];

static float absf_local(float value)
{
return value < 0.0f ? -value : value;
}

static float input_value(uint32_t i)
{
return ((float)((i * 13u + 7u) % 97u) * 0.0625f) - 3.0f;
}

static void fill_case(uint32_t n)
{
for (uint32_t i = 0; i < CONV1D_3TAP_MAX_N; i++)
x_values[i] = input_value(i);

for (uint32_t i = 0; i < CONV1D_3TAP_MAX_N + HOST_GUARD_FLOATS; i++)
    out_values[i] = HOST_SENTINEL;

for (uint32_t i = 0; i < CONV1D_3TAP_MAX_N; i++)
    expected_values[i] = 0.0f;

(void)n;

}

static void run_cpu_reference(uint32_t n, float c0, float c1, float c2)
{
for (uint32_t i = 0; i < n; i++)
{
uint32_t left = (i == 0) ? 0 : (i - 1u);
uint32_t right = (i + 1u >= n) ? (n - 1u) : (i + 1u);
expected_values[i] =
c0 * x_values[left] +
c1 * x_values[i] +
c2 * x_values[right];
}
}

static int scaled_checksum(const float *values, uint32_t n)
{
int checksum = 0;
for (uint32_t i = 0; i < n; i++)
checksum += (int)(values[i] * CHECKSUM_SCALE);
return checksum;
}

static void verify_results(
uint32_t case_id,
uint32_t n,
int *mismatch_count,
float *max_abs_diff)
{
*mismatch_count = 0;
*max_abs_diff = 0.0f;

for (uint32_t i = 0; i < n; i++)
{
    float diff = out_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
        *max_abs_diff = abs_diff;

    if (abs_diff > CONV1D_3TAP_EPSILON)
    {
        if (*mismatch_count < 8)
        {
            printk("ERROR: case=%d n=%d i=%d gpu=%f cpu=%f diff=%f\n",
                   (int)case_id,
                   (int)n,
                   (int)i,
                   out_values[i],
                   expected_values[i],
                   diff);
        }
        (*mismatch_count)++;
    }
}

}

static int verify_host_guard(uint32_t n)
{
int mismatches = 0;

for (uint32_t i = n;
     i < n + HOST_GUARD_FLOATS && i < CONV1D_3TAP_MAX_N + HOST_GUARD_FLOATS;
     i++)
{
    if (out_values[i] != HOST_SENTINEL)
    {
        if (mismatches < 8)
        {
            printk("ERROR: host guard changed n=%d i=%d value=%f expected=%f\n",
                   (int)n,
                   (int)i,
                   out_values[i],
                   HOST_SENTINEL);
        }
        mismatches++;
    }
}

return mismatches;

}

void notmain(void)
{
struct vc4_runtime rt;
struct conv1d_3tap_state state;
const float c0 = -0.25f;
const float c1 = 1.5f;
const float c2 = 0.75f;

if (conv1d_3tap_prepare(&rt, &state, CONV1D_3TAP_MAX_N) < 0)
    panic("conv1d_3tap runtime setup failed");

uint32_t activeQpus = state.active_qpus;
uint32_t laneWidth = state.lane_width;

if (activeQpus != VC4_RUNTIME_MAX_QPUS)
    panic("Unexpected active QPU count: %d", (int)activeQpus);
if (laneWidth != VC4_RUNTIME_LANE_WIDTH)
    panic("Unexpected lane width: %d", (int)laneWidth);

printk("Running VC4 conv1d_3tap reference bundle...\n");
printk("CONV1D_3TAP_RUNTIME_SETUP max_n=%d allocations=%d\n",
       CONV1D_3TAP_MAX_N,
       (int)state.runtime_allocations);

int start = timer_get_usec();
int totalMismatches = 0;
int sentinelMismatches = 0;
int launchFailures = 0;
int checksumAccum = 0;
float maxAbsDiffOverall = 0.0f;

for (uint32_t caseId = 0; caseId < CONV1D_3TAP_CASES; caseId++)
{
    uint32_t n = test_sizes[caseId];

    fill_case(n);
    run_cpu_reference(n, c0, c1, c2);

    if (conv1d_3tap_launch(&state, x_values, out_values, n, c0, c1, c2) < 0)
    {
        printk("ERROR: conv1d_3tap launch failed for case=%d n=%d\n",
               (int)caseId,
               (int)n);
        launchFailures++;
        continue;
    }

    int mismatches = 0;
    float maxAbsDiff = 0.0f;
    verify_results(caseId, n, &mismatches, &maxAbsDiff);

    int caseSentinelMismatches =
        verify_host_guard(n) + (int)state.last_sentinel_mismatches;
    int checksum = scaled_checksum(out_values, n);
    int expectedChecksum = scaled_checksum(expected_values, n);

    if (checksum != expectedChecksum)
    {
        printk("ERROR: checksum mismatch case=%d n=%d gpu=%d cpu=%d\n",
               (int)caseId,
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

    printk("CONV1D_3TAP_CASE case=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
           (int)caseId,
           (int)n,
           mismatches,
           caseSentinelMismatches,
           checksum,
           maxAbsDiff,
           (int)state.runtime_launches,
           (int)state.runtime_allocations);
}

int end = timer_get_usec();
int elapsed = end - start;

const char *status =
    (totalMismatches == 0 &&
     sentinelMismatches == 0 &&
     launchFailures == 0 &&
     activeQpus == VC4_RUNTIME_MAX_QPUS &&
     laneWidth == VC4_RUNTIME_LANE_WIDTH &&
     state.max_n == CONV1D_3TAP_MAX_N &&
     state.runtime_allocations == 1u &&
     state.runtime_launches == CONV1D_3TAP_CASES) ? "PASS" : "FAIL";

printk("VC4_TEST_RESULT name=conv1d_3tap status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d checksum_accum=%d elapsed_usec=%d\n",
       status,
       CONV1D_3TAP_CASES,
       totalMismatches,
       sentinelMismatches,
       launchFailures,
       (int)activeQpus,
       (int)laneWidth,
       CONV1D_3TAP_MAX_N,
       maxAbsDiffOverall,
       (int)state.runtime_allocations,
       (int)state.runtime_launches,
       checksumAccum,
       elapsed);

conv1d_3tap_shutdown(&state);

}

