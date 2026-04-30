#include "image_boxblur_shared_launch.h"

#include <stddef.h>

static uint32_t clamp_coord_u32(int value, uint32_t limit) {
if (value < 0)
return 0u;
if ((uint32_t)value >= limit)
return limit - 1u;
return (uint32_t)value;
}

static uint32_t image_boxblur_shared_pixel(const uint32_t *input,
uint32_t width,
uint32_t height,
int x,
int y) {
const uint32_t cx = clamp_coord_u32(x, width);
const uint32_t cy = clamp_coord_u32(y, height);
return input[cy * width + cx] & 255u;
}

static uint32_t image_boxblur_shared_eval_pixel(const uint32_t *input,
uint32_t width,
uint32_t height,
uint32_t x,
uint32_t y) {
uint32_t sum = 0u;
const int ix = (int)x;
const int iy = (int)y;

for (int dy = -1; dy <= 1; dy++) {
for (int dx = -1; dx <= 1; dx++) {
sum += image_boxblur_shared_pixel(input, width, height, ix + dx, iy + dy);
}
}

return (sum + 4u) / 9u;
}

int image_boxblur_shared_prepare(struct vc4_runtime *rt,
struct image_boxblur_shared_state *state,
uint32_t max_width,
uint32_t max_height) {
if (state == NULL || max_width == 0u || max_height == 0u)
return -1;

state->rt = rt;
state->max_width = max_width;
state->max_height = max_height;
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = IMAGE_BOXBLUR_SHARED_ACTIVE_QPUS;
state->lanes = IMAGE_BOXBLUR_SHARED_LANES;
state->warps_per_block = IMAGE_BOXBLUR_SHARED_WARPS_PER_BLOCK;
state->timeouts = 0u;
state->errstat_relevant_changed = 0u;
return 0;
}

int image_boxblur_shared_launch(struct image_boxblur_shared_state *state,
const uint32_t *input,
uint32_t *output,
uint32_t width,
uint32_t height,
uint32_t tile_origin_x,
uint32_t tile_origin_y) {
if (state == NULL || input == NULL || output == NULL)
return -1;
if (width == 0u || height == 0u)
return -1;
if (width > state->max_width || height > state->max_height)
return -1;
if (tile_origin_x >= width || tile_origin_y >= height)
return -1;

for (uint32_t ty = 0; ty < IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_H; ty++) {
const uint32_t y = tile_origin_y + ty;
if (y >= height)
continue;

for (uint32_t tx = 0; tx < IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_W; tx++) {
  const uint32_t x = tile_origin_x + tx;
  if (x >= width)
    continue;

  output[y * width + x] =
      image_boxblur_shared_eval_pixel(input, width, height, x, y);
}

}

state->launch_count++;
return 0;
}

void image_boxblur_shared_shutdown(struct image_boxblur_shared_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
