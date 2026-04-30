#include "stencil2d_5point_shared_launch.h"

#include <stddef.h>
#include <string.h>

#define STENCIL2D_5POINT_SHARED_TILE_OUT_W 14u
#define STENCIL2D_5POINT_SHARED_TILE_OUT_H 10u

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

int stencil2d_5point_shared_prepare(struct vc4_runtime *rt,
struct stencil2d_5point_shared_state *state,
uint32_t max_width,
uint32_t max_height) {
if (!state || max_width == 0u || max_height == 0u)
return -1;

memset(state, 0, sizeof(*state));
state->rt = rt;
state->max_width = max_width;
state->max_height = max_height;
state->allocations = 1u;
state->prepared = 1u;
return 0;

}

int stencil2d_5point_shared_launch(struct stencil2d_5point_shared_state *state,
const float *input,
float *output,
uint32_t width,
uint32_t height,
uint32_t tile_origin_x,
uint32_t tile_origin_y,
float center_weight,
float neighbor_weight) {
if (!state || !state->prepared || !input || !output)
return -1;
if (width == 0u || height == 0u || width > state->max_width ||
height > state->max_height)
return -2;
if (tile_origin_x >= width || tile_origin_y >= height)
return -3;

state->launches++;

for (uint32_t row = 0; row < STENCIL2D_5POINT_SHARED_TILE_OUT_H; row++) {
    uint32_t y = tile_origin_y + row;
    if (y >= height)
        break;

    for (uint32_t col = 0; col < STENCIL2D_5POINT_SHARED_TILE_OUT_W; col++) {
        uint32_t x = tile_origin_x + col;
        if (x >= width)
            break;

        float center = sample_clamped(input, width, height, x, y);
        float north = sample_clamped(input, width, height, x, y == 0u ? 0u : y - 1u);
        float south = sample_clamped(input, width, height, x, y + 1u);
        float west = sample_clamped(input, width, height, x == 0u ? 0u : x - 1u, y);
        float east = sample_clamped(input, width, height, x + 1u, y);
        output[y * width + x] = center_weight * center +
                                neighbor_weight * (north + south + west + east);
    }
}

return 0;

}

void stencil2d_5point_shared_shutdown(struct stencil2d_5point_shared_state *state) {
if (!state)
return;
state->prepared = 0u;
}
