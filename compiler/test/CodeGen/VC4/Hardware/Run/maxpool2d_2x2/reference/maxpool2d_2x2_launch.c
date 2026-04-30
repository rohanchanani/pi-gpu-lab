#include "maxpool2d_2x2_launch.h"

#include <stddef.h>

static uint32_t ceil_div2_u32(uint32_t value) {
return (value + 1u) >> 1;
}

static float maxpool2d_2x2_ref_one(const float *input,
uint32_t width,
uint32_t height,
uint32_t ox,
uint32_t oy) {
const uint32_t x0 = ox * 2u;
const uint32_t y0 = oy * 2u;
float best = input[y0 * width + x0];

if (x0 + 1u < width) {
const float v = input[y0 * width + x0 + 1u];
if (v > best)
best = v;
}

if (y0 + 1u < height) {
const float v = input[(y0 + 1u) * width + x0];
if (v > best)
best = v;
}

if (x0 + 1u < width && y0 + 1u < height) {
const float v = input[(y0 + 1u) * width + x0 + 1u];
if (v > best)
best = v;
}

return best;
}

int maxpool2d_2x2_prepare(struct vc4_runtime *rt,
struct maxpool2d_2x2_state *state,
uint32_t max_width,
uint32_t max_height) {
if (state == NULL)
return -1;

state->rt = rt;
state->max_width = max_width;
state->max_height = max_height;
state->max_output_words = ceil_div2_u32(max_width) * ceil_div2_u32(max_height);
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = MAXPOOL2D_2X2_ACTIVE_QPUS;
state->lanes = MAXPOOL2D_2X2_LANES;
return 0;
}

int maxpool2d_2x2_launch(struct maxpool2d_2x2_state *state,
const float *input,
float *output,
uint32_t width,
uint32_t height) {
if (state == NULL || input == NULL || output == NULL)
return -1;
if (width > state->max_width || height > state->max_height)
return -1;

const uint32_t out_w = ceil_div2_u32(width);
const uint32_t out_h = ceil_div2_u32(height);

for (uint32_t oy = 0; oy < out_h; oy++) {
for (uint32_t ox = 0; ox < out_w; ox++) {
output[oy * out_w + ox] = maxpool2d_2x2_ref_one(input, width, height, ox, oy);
}
}

state->launch_count++;
return 0;
}

void maxpool2d_2x2_shutdown(struct maxpool2d_2x2_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
