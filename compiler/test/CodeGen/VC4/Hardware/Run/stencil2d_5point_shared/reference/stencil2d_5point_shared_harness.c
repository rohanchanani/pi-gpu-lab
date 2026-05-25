#include "rpi.h"
#include "stencil2d_5point_shared_launch.h"

#define STENCIL2D_5POINT_SHARED_TILE_OUT_W 14u
#define STENCIL2D_5POINT_SHARED_TILE_OUT_H 10u
#define STENCIL2D_5POINT_SHARED_CASES 3u
#define STENCIL2D_5POINT_SHARED_SENTINEL (-123456.0f)
#define STENCIL2D_5POINT_SHARED_EPSILON 0.001f
#define CHECKSUM_SCALE 4096.0f

struct stencil_case
{
    uint32_t width;
    uint32_t height;
    uint32_t origin_x;
    uint32_t origin_y;
};

static const struct stencil_case test_cases[STENCIL2D_5POINT_SHARED_CASES] = {
    {18u, 14u, 0u, 0u},
    {31u, 19u, 3u, 2u},
    {31u, 19u, 17u, 9u},
};

static float input_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];
static float output_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];
static float expected_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];

static float absf_local(float value)
{
    return value < 0.0f ? -value : value;
}

static int invalid_f32(float value)
{
    return !(value == value) || value > 3.4e38f || value < -3.4e38f;
}

static float input_value(uint32_t x, uint32_t y)
{
    uint32_t m = (x * 13u + y * 17u + x * y + 5u) % 97u;
    return ((float)m) * 0.25f - 8.0f;
}

static uint32_t clamp_u32(uint32_t value, uint32_t limit)
{
    if (limit == 0u)
        return 0u;
    return value < limit ? value : limit - 1u;
}

static float sample_clamped(const float *input, uint32_t width, uint32_t height, uint32_t x, uint32_t y)
{
    x = clamp_u32(x, width);
    y = clamp_u32(y, height);
    return input[y * width + x];
}

static int in_output_tile(uint32_t x, uint32_t y, uint32_t origin_x, uint32_t origin_y)
{
    return x >= origin_x && x < origin_x + STENCIL2D_5POINT_SHARED_TILE_OUT_W &&
           y >= origin_y && y < origin_y + STENCIL2D_5POINT_SHARED_TILE_OUT_H;
}

static void fill_case(uint32_t width,
                      uint32_t height,
                      uint32_t origin_x,
                      uint32_t origin_y,
                      float center_weight,
                      float neighbor_weight)
{
    for (uint32_t i = 0; i < STENCIL2D_5POINT_SHARED_MAX_WORDS; i++)
    {
        input_values[i] = 0.0f;
        output_values[i] = STENCIL2D_5POINT_SHARED_SENTINEL;
        expected_values[i] = STENCIL2D_5POINT_SHARED_SENTINEL;
    }

    for (uint32_t y = 0; y < height; y++)
    {
        for (uint32_t x = 0; x < width; x++)
            input_values[y * width + x] = input_value(x, y);
    }

    for (uint32_t row = 0; row < STENCIL2D_5POINT_SHARED_TILE_OUT_H; row++)
    {
        uint32_t y = origin_y + row;
        if (y >= height)
            break;

        for (uint32_t col = 0; col < STENCIL2D_5POINT_SHARED_TILE_OUT_W; col++)
        {
            uint32_t x = origin_x + col;
            if (x >= width)
                break;

            float center = sample_clamped(input_values, width, height, x, y);
            float north = sample_clamped(input_values, width, height, x, y == 0u ? 0u : y - 1u);
            float south = sample_clamped(input_values, width, height, x, y + 1u);
            float west = sample_clamped(input_values, width, height, x == 0u ? 0u : x - 1u, y);
            float east = sample_clamped(input_values, width, height, x + 1u, y);
            expected_values[y * width + x] =
                center_weight * center + neighbor_weight * (north + south + west + east);
        }
    }
}

static int scaled_checksum(uint32_t width, uint32_t height, uint32_t origin_x, uint32_t origin_y)
{
    int checksum = 0;
    for (uint32_t y = 0; y < height; y++)
    {
        for (uint32_t x = 0; x < width; x++)
        {
            if (in_output_tile(x, y, origin_x, origin_y))
                checksum += (int)(output_values[y * width + x] * CHECKSUM_SCALE);
        }
    }
    return checksum;
}

void notmain(void)
{
    struct vc4_runtime rt;
    struct stencil2d_5point_shared_state state;
    const float centerWeight = 0.5f;
    const float neighborWeight = 0.125f;

    if (stencil2d_5point_shared_prepare(&rt,
                                        &state,
                                        STENCIL2D_5POINT_SHARED_MAX_WIDTH,
                                        STENCIL2D_5POINT_SHARED_MAX_HEIGHT) < 0)
        panic("stencil2d_5point_shared runtime setup failed");

    uint32_t activeQpus = state.active_qpus;
    uint32_t laneWidth = state.lane_width;

    if (activeQpus != VC4_RUNTIME_MAX_QPUS)
        panic("Unexpected active QPU count: %d", (int)activeQpus);
    if (laneWidth != VC4_RUNTIME_LANE_WIDTH)
        panic("Unexpected lane width: %d", (int)laneWidth);

    printk("Running VC4 stencil2d_5point_shared reference bundle...\n");
    printk("STENCIL2D_5POINT_SHARED_RUNTIME_SETUP max_width=%d max_height=%d allocations=%d warps_per_block=%d\n",
           STENCIL2D_5POINT_SHARED_MAX_WIDTH,
           STENCIL2D_5POINT_SHARED_MAX_HEIGHT,
           (int)state.runtime_allocations,
           VC4_RUNTIME_MAX_QPUS);

    int start = timer_get_usec();
    int totalMismatches = 0;
    int sentinelMismatches = 0;
    int launchFailures = 0;
    int checkedElements = 0;
    int checksumAccum = 0;
    float maxAbsDiffOverall = 0.0f;

    for (uint32_t caseId = 0; caseId < STENCIL2D_5POINT_SHARED_CASES; caseId++)
    {
        const struct stencil_case *tc = &test_cases[caseId];

        fill_case(tc->width,
                  tc->height,
                  tc->origin_x,
                  tc->origin_y,
                  centerWeight,
                  neighborWeight);

        if (stencil2d_5point_shared_launch(&state,
                                           input_values,
                                           output_values,
                                           tc->width,
                                           tc->height,
                                           tc->origin_x,
                                           tc->origin_y,
                                           centerWeight,
                                           neighborWeight) < 0)
        {
            printk("ERROR: stencil2d_5point_shared launch failed for case=%d width=%d height=%d origin_x=%d origin_y=%d\n",
                   (int)caseId,
                   (int)tc->width,
                   (int)tc->height,
                   (int)tc->origin_x,
                   (int)tc->origin_y);
            launchFailures++;
            continue;
        }

        int mismatches = 0;
        int caseSentinelMismatches = (int)state.last_sentinel_mismatches;
        int caseChecked = 0;
        float maxAbsDiff = 0.0f;

        for (uint32_t y = 0; y < tc->height; y++)
        {
            for (uint32_t x = 0; x < tc->width; x++)
            {
                uint32_t i = y * tc->width + x;
                if (in_output_tile(x, y, tc->origin_x, tc->origin_y))
                {
                    float diff = output_values[i] - expected_values[i];
                    float absDiff = absf_local(diff);
                    caseChecked++;

                    if (absDiff > maxAbsDiff)
                        maxAbsDiff = absDiff;

                    if (invalid_f32(output_values[i]) || absDiff > STENCIL2D_5POINT_SHARED_EPSILON)
                    {
                        if (mismatches < 8)
                        {
                            printk("ERROR: case=%d width=%d height=%d y=%d x=%d gpu=%f cpu=%f diff=%f\n",
                                   (int)caseId,
                                   (int)tc->width,
                                   (int)tc->height,
                                   (int)y,
                                   (int)x,
                                   output_values[i],
                                   expected_values[i],
                                   diff);
                        }
                        mismatches++;
                    }
                }
                else if (output_values[i] != STENCIL2D_5POINT_SHARED_SENTINEL)
                {
                    if (caseSentinelMismatches < 8)
                    {
                        printk("ERROR: sentinel changed case=%d y=%d x=%d value=%f\n",
                               (int)caseId,
                               (int)y,
                               (int)x,
                               output_values[i]);
                    }
                    caseSentinelMismatches++;
                }
            }
        }

        int checksum = scaled_checksum(tc->width, tc->height, tc->origin_x, tc->origin_y);

        if (maxAbsDiff > maxAbsDiffOverall)
            maxAbsDiffOverall = maxAbsDiff;

        totalMismatches += mismatches;
        sentinelMismatches += caseSentinelMismatches;
        checkedElements += caseChecked;
        checksumAccum += checksum;

        printk("STENCIL2D_5POINT_SHARED_CASE case=%d width=%d height=%d origin_x=%d origin_y=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
               (int)caseId,
               (int)tc->width,
               (int)tc->height,
               (int)tc->origin_x,
               (int)tc->origin_y,
               caseChecked,
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
         checkedElements > 0 &&
         maxAbsDiffOverall <= STENCIL2D_5POINT_SHARED_EPSILON &&
         activeQpus == VC4_RUNTIME_MAX_QPUS &&
         laneWidth == VC4_RUNTIME_LANE_WIDTH &&
         state.runtime_allocations == 1u &&
         state.runtime_launches == STENCIL2D_5POINT_SHARED_CASES) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=stencil2d_5point_shared status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d warps_per_block=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d timeouts=%d errstat_relevant_changed=%d elapsed_usec=%d\n",
           status,
           STENCIL2D_5POINT_SHARED_CASES,
           checkedElements,
           totalMismatches,
           sentinelMismatches,
           launchFailures,
           (int)activeQpus,
           (int)laneWidth,
           VC4_RUNTIME_MAX_QPUS,
           checksumAccum,
           maxAbsDiffOverall,
           (int)state.runtime_allocations,
           (int)state.runtime_launches,
           0,
           0,
           elapsed);

    stencil2d_5point_shared_shutdown(&state);
}
