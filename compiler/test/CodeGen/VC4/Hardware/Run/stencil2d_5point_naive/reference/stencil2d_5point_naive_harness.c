#include "rpi.h"
#include "stencil2d_5point_naive_launch.h"

#define STENCIL2D_5POINT_NAIVE_CASES 9u
#define STENCIL2D_5POINT_NAIVE_EPSILON 0.0005f
#define CHECKSUM_SCALE 4096.0f
#define HOST_GUARD_FLOATS 32u
#define HOST_SENTINEL (-13579.0f)

struct stencil_case
{
uint32_t width;
uint32_t height;
};

static const struct stencil_case test_cases[STENCIL2D_5POINT_NAIVE_CASES] = {
{0u, 7u},
{1u, 1u},
{2u, 3u},
{5u, 7u},
{7u, 5u},
{16u, 4u},
{17u, 9u},
{31u, 19u},
{32u, 32u},
};

static float input_values[STENCIL2D_5POINT_NAIVE_MAX_WORDS];
static float output_values[STENCIL2D_5POINT_NAIVE_MAX_WORDS + HOST_GUARD_FLOATS];
static float expected_values[STENCIL2D_5POINT_NAIVE_MAX_WORDS];

static float absf_local(float value)
{
return value < 0.0f ? -value : value;
}

static uint32_t compact_index(uint32_t width, uint32_t y, uint32_t x)
{
return y * width + x;
}

static float input_value(uint32_t i, uint32_t x, uint32_t y)
{
int raw = (int)((i * 13u + x * 7u + y * 5u + 11u) % 113u) - 56;
return (float)raw * 0.03125f;
}

static void fill_case(uint32_t width, uint32_t height)
{
uint32_t total = width * height;

for (uint32_t i = 0; i < STENCIL2D_5POINT_NAIVE_MAX_WORDS; i++)
{
    input_values[i] = 0.0f;
    expected_values[i] = 0.0f;
}

for (uint32_t y = 0; y < height; y++)
{
    for (uint32_t x = 0; x < width; x++)
    {
        uint32_t i = compact_index(width, y, x);
        input_values[i] = input_value(i, x, y);
    }
}

for (uint32_t i = 0; i < STENCIL2D_5POINT_NAIVE_MAX_WORDS + HOST_GUARD_FLOATS; i++)
    output_values[i] = HOST_SENTINEL;

(void)total;

}

static void run_cpu_reference(
uint32_t width,
uint32_t height,
float center_weight,
float neighbor_weight)
{
if (width == 0 || height == 0)
return;

for (uint32_t y = 0; y < height; y++)
{
    uint32_t up_y = (y == 0) ? 0 : (y - 1u);
    uint32_t down_y = (y + 1u >= height) ? (height - 1u) : (y + 1u);

    for (uint32_t x = 0; x < width; x++)
    {
        uint32_t left_x = (x == 0) ? 0 : (x - 1u);
        uint32_t right_x = (x + 1u >= width) ? (width - 1u) : (x + 1u);

        float center = input_values[compact_index(width, y, x)];
        float up = input_values[compact_index(width, up_y, x)];
        float down = input_values[compact_index(width, down_y, x)];
        float left = input_values[compact_index(width, y, left_x)];
        float right = input_values[compact_index(width, y, right_x)];

        expected_values[compact_index(width, y, x)] =
            center_weight * center +
            neighbor_weight * (up + down + left + right);
    }
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
uint32_t width,
uint32_t height,
int *mismatch_count,
float *max_abs_diff)
{
uint32_t total = width * height;
*mismatch_count = 0;
*max_abs_diff = 0.0f;

for (uint32_t i = 0; i < total; i++)
{
    float diff = output_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
        *max_abs_diff = abs_diff;

    if (abs_diff > STENCIL2D_5POINT_NAIVE_EPSILON)
    {
        if (*mismatch_count < 8)
        {
            uint32_t y = width ? (i / width) : 0;
            uint32_t x = width ? (i % width) : 0;
            printk("ERROR: case=%d width=%d height=%d y=%d x=%d gpu=%f cpu=%f diff=%f\n",
                   (int)case_id,
                   (int)width,
                   (int)height,
                   (int)y,
                   (int)x,
                   output_values[i],
                   expected_values[i],
                   diff);
        }
        (*mismatch_count)++;
    }
}

}

static int verify_host_guard(uint32_t total)
{
int mismatches = 0;

for (uint32_t i = total;
     i < total + HOST_GUARD_FLOATS &&
     i < STENCIL2D_5POINT_NAIVE_MAX_WORDS + HOST_GUARD_FLOATS;
     i++)
{
    if (output_values[i] != HOST_SENTINEL)
    {
        if (mismatches < 8)
        {
            printk("ERROR: host guard changed total=%d i=%d value=%f expected=%f\n",
                   (int)total,
                   (int)i,
                   output_values[i],
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
struct stencil2d_5point_naive_state state;
const float centerWeight = 0.5f;
const float neighborWeight = 0.125f;

if (stencil2d_5point_naive_prepare(&rt,
                                    &state,
                                    STENCIL2D_5POINT_NAIVE_MAX_WIDTH,
                                    STENCIL2D_5POINT_NAIVE_MAX_HEIGHT) < 0)
    panic("stencil2d_5point_naive runtime setup failed");

uint32_t activeQpus = state.active_qpus;
uint32_t laneWidth = state.lane_width;

if (activeQpus != VC4_RUNTIME_MAX_QPUS)
    panic("Unexpected active QPU count: %d", (int)activeQpus);
if (laneWidth != VC4_RUNTIME_LANE_WIDTH)
    panic("Unexpected lane width: %d", (int)laneWidth);

printk("Running VC4 stencil2d_5point_naive reference bundle...\n");
printk("STENCIL2D_5POINT_NAIVE_RUNTIME_SETUP max_width=%d max_height=%d allocations=%d\n",
       STENCIL2D_5POINT_NAIVE_MAX_WIDTH,
       STENCIL2D_5POINT_NAIVE_MAX_HEIGHT,
       (int)state.runtime_allocations);

int start = timer_get_usec();
int totalMismatches = 0;
int sentinelMismatches = 0;
int launchFailures = 0;
int checksumAccum = 0;
float maxAbsDiffOverall = 0.0f;

for (uint32_t caseId = 0; caseId < STENCIL2D_5POINT_NAIVE_CASES; caseId++)
{
    uint32_t width = test_cases[caseId].width;
    uint32_t height = test_cases[caseId].height;
    uint32_t total = width * height;

    fill_case(width, height);
    run_cpu_reference(width, height, centerWeight, neighborWeight);

    if (stencil2d_5point_naive_launch(&state,
                                       input_values,
                                       output_values,
                                       width,
                                       height,
                                       centerWeight,
                                       neighborWeight) < 0)
    {
        printk("ERROR: stencil2d_5point_naive launch failed for case=%d width=%d height=%d\n",
               (int)caseId,
               (int)width,
               (int)height);
        launchFailures++;
        continue;
    }

    int mismatches = 0;
    float maxAbsDiff = 0.0f;
    verify_results(caseId, width, height, &mismatches, &maxAbsDiff);

    int caseSentinelMismatches =
        verify_host_guard(total) + (int)state.last_sentinel_mismatches;
    int checksum = scaled_checksum(output_values, total);
    int expectedChecksum = scaled_checksum(expected_values, total);

    if (checksum != expectedChecksum)
    {
        printk("ERROR: checksum mismatch case=%d width=%d height=%d gpu=%d cpu=%d\n",
               (int)caseId,
               (int)width,
               (int)height,
               checksum,
               expectedChecksum);
        mismatches++;
    }

    if (maxAbsDiff > maxAbsDiffOverall)
        maxAbsDiffOverall = maxAbsDiff;

    totalMismatches += mismatches;
    sentinelMismatches += caseSentinelMismatches;
    checksumAccum += checksum;

    printk("STENCIL2D_5POINT_NAIVE_CASE case=%d width=%d height=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
           (int)caseId,
           (int)width,
           (int)height,
           (int)total,
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
     state.max_width == STENCIL2D_5POINT_NAIVE_MAX_WIDTH &&
     state.max_height == STENCIL2D_5POINT_NAIVE_MAX_HEIGHT &&
     state.runtime_allocations == 1u &&
     state.runtime_launches == STENCIL2D_5POINT_NAIVE_CASES) ? "PASS" : "FAIL";

printk("VC4_TEST_RESULT name=stencil2d_5point_naive status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_width=%d max_height=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d checksum_accum=%d elapsed_usec=%d\n",
       status,
       STENCIL2D_5POINT_NAIVE_CASES,
       totalMismatches,
       sentinelMismatches,
       launchFailures,
       (int)activeQpus,
       (int)laneWidth,
       STENCIL2D_5POINT_NAIVE_MAX_WIDTH,
       STENCIL2D_5POINT_NAIVE_MAX_HEIGHT,
       maxAbsDiffOverall,
       (int)state.runtime_allocations,
       (int)state.runtime_launches,
       checksumAccum,
       elapsed);

stencil2d_5point_naive_shutdown(&state);

}

