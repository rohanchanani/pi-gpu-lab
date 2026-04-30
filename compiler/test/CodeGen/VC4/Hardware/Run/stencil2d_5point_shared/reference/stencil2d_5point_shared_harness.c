#include "stencil2d_5point_shared_launch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define STENCIL2D_5POINT_SHARED_MAX_WIDTH 31u
#define STENCIL2D_5POINT_SHARED_MAX_HEIGHT 19u
#define STENCIL2D_5POINT_SHARED_MAX_WORDS
(STENCIL2D_5POINT_SHARED_MAX_WIDTH * STENCIL2D_5POINT_SHARED_MAX_HEIGHT)
#define STENCIL2D_5POINT_SHARED_TILE_OUT_W 14u
#define STENCIL2D_5POINT_SHARED_TILE_OUT_H 10u
#define STENCIL2D_5POINT_SHARED_CASES 3u
#define STENCIL2D_5POINT_SHARED_SENTINEL (-123456.0f)

struct stencil_case {
uint32_t width;
uint32_t height;
uint32_t origin_x;
uint32_t origin_y;
};

static const struct stencil_case cases[STENCIL2D_5POINT_SHARED_CASES] = {
{18u, 14u, 0u, 0u},
{31u, 19u, 3u, 2u},
{31u, 19u, 17u, 9u},
};

static float input_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];
static float output_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];
static float expected_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];

static float deterministic_input(uint32_t x, uint32_t y) {
uint32_t m = (x * 13u + y * 17u + x * y + 5u) % 97u;
return ((float)m) * 0.25f - 8.0f;
}

static uint32_t clamp_u32_to_range(uint32_t value, uint32_t limit) {
if (limit == 0u)
return 0u;
return value < limit ? value : limit - 1u;
}

static float sample_clamped(const float *input, uint32_t width, uint32_t height,
uint32_t x, uint32_t y) {
x = clamp_u32_to_range(x, width);
y = clamp_u32_to_range(y, height);
return input[y * width + x];
}

static void fill_input(uint32_t width, uint32_t height) {
for (uint32_t y = 0; y < height; y++) {
for (uint32_t x = 0; x < width; x++) {
input_values[y * width + x] = deterministic_input(x, y);
}
}

for (uint32_t i = 0; i < STENCIL2D_5POINT_SHARED_MAX_WORDS; i++) {
    output_values[i] = STENCIL2D_5POINT_SHARED_SENTINEL;
    expected_values[i] = STENCIL2D_5POINT_SHARED_SENTINEL;
}

}

static void fill_expected(uint32_t width, uint32_t height,
uint32_t origin_x, uint32_t origin_y,
float center_weight, float neighbor_weight) {
for (uint32_t row = 0; row < STENCIL2D_5POINT_SHARED_TILE_OUT_H; row++) {
uint32_t y = origin_y + row;
if (y >= height)
break;

    for (uint32_t col = 0; col < STENCIL2D_5POINT_SHARED_TILE_OUT_W; col++) {
        uint32_t x = origin_x + col;
        if (x >= width)
            break;

        float center = sample_clamped(input_values, width, height, x, y);
        float north = sample_clamped(input_values, width, height, x, y == 0u ? 0u : y - 1u);
        float south = sample_clamped(input_values, width, height, x, y + 1u);
        float west = sample_clamped(input_values, width, height, x == 0u ? 0u : x - 1u, y);
        float east = sample_clamped(input_values, width, height, x + 1u, y);
        expected_values[y * width + x] = center_weight * center +
                                         neighbor_weight * (north + south + west + east);
    }
}

}

static int in_output_tile(uint32_t x, uint32_t y,
uint32_t origin_x, uint32_t origin_y) {
return x >= origin_x &&
x < origin_x + STENCIL2D_5POINT_SHARED_TILE_OUT_W &&
y >= origin_y &&
y < origin_y + STENCIL2D_5POINT_SHARED_TILE_OUT_H;
}

static int32_t checksum_scaled(uint32_t width, uint32_t height,
uint32_t origin_x, uint32_t origin_y) {
int32_t checksum = 0;
for (uint32_t y = 0; y < height; y++) {
for (uint32_t x = 0; x < width; x++) {
if (!in_output_tile(x, y, origin_x, origin_y))
continue;
float v = output_values[y * width + x];
checksum += (int32_t)(v * 4096.0f);
}
}
return checksum;
}

int main(void) {
const float center_weight = 0.5f;
const float neighbor_weight = 0.125f;
struct stencil2d_5point_shared_state state;

int launch_failures = 0;
int total_mismatches = 0;
int total_sentinel_mismatches = 0;
float max_abs_diff = 0.0f;

if (stencil2d_5point_shared_prepare(0, &state,
                                    STENCIL2D_5POINT_SHARED_MAX_WIDTH,
                                    STENCIL2D_5POINT_SHARED_MAX_HEIGHT) != 0) {
    printf("VC4_TEST_RESULT name=stencil2d_5point_shared status=FAIL cases=3 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 warps_per_block=12 max_abs_diff=0.0 runtime_allocations=0 runtime_launches=0 timeouts=0 errstat_relevant_changed=0 elapsed_usec=0\n");
    return 1;
}

printf("STENCIL2D_5POINT_SHARED_RUNTIME_SETUP max_width=31 max_height=19 allocations=1 warps_per_block=12\n");

for (uint32_t case_id = 0; case_id < STENCIL2D_5POINT_SHARED_CASES; case_id++) {
    const struct stencil_case *tc = &cases[case_id];
    int mismatches = 0;
    int sentinel_mismatches = 0;

    fill_input(tc->width, tc->height);
    fill_expected(tc->width, tc->height, tc->origin_x, tc->origin_y,
                  center_weight, neighbor_weight);

    if (stencil2d_5point_shared_launch(&state, input_values, output_values,
                                       tc->width, tc->height,
                                       tc->origin_x, tc->origin_y,
                                       center_weight, neighbor_weight) != 0) {
        launch_failures++;
    }

    for (uint32_t y = 0; y < tc->height; y++) {
        for (uint32_t x = 0; x < tc->width; x++) {
            uint32_t i = y * tc->width + x;
            if (in_output_tile(x, y, tc->origin_x, tc->origin_y)) {
                float diff = output_values[i] - expected_values[i];
                float adiff = fabsf(diff);
                if (adiff > max_abs_diff)
                    max_abs_diff = adiff;
                if (adiff > 0.001f)
                    mismatches++;
            } else if (output_values[i] != STENCIL2D_5POINT_SHARED_SENTINEL) {
                sentinel_mismatches++;
            }
        }
    }

    total_mismatches += mismatches;
    total_sentinel_mismatches += sentinel_mismatches;

    printf("STENCIL2D_5POINT_SHARED_CASE case=%u width=%u height=%u origin_x=%u origin_y=%u mismatches=%d sentinel_mismatches=%d checksum=0x%08x max_abs_diff=%0.6f launches=%u allocations=1\n",
           case_id, tc->width, tc->height, tc->origin_x, tc->origin_y,
           mismatches, sentinel_mismatches,
           (uint32_t)checksum_scaled(tc->width, tc->height, tc->origin_x, tc->origin_y),
           max_abs_diff, state.launches);
}

const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                      total_sentinel_mismatches == 0) ? "PASS" : "FAIL";

printf("VC4_TEST_RESULT name=stencil2d_5point_shared status=%s cases=3 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=12 lanes=16 warps_per_block=12 max_abs_diff=%0.6f runtime_allocations=1 runtime_launches=%u timeouts=0 errstat_relevant_changed=0 elapsed_usec=0\n",
       status, total_mismatches, total_sentinel_mismatches, launch_failures,
       max_abs_diff, state.launches);

stencil2d_5point_shared_shutdown(&state);
return status[0] == 'P' ? 0 : 1;

}
