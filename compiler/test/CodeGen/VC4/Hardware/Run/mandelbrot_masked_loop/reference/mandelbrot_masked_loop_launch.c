#include "mandelbrot_masked_loop_launch.h"

#include <stddef.h>

static uint32_t mandelbrot_pixel_count(float cx, float cy, uint32_t max_iter) {
float zx = 0.0f;
float zy = 0.0f;
uint32_t iter = 0u;

while (iter < max_iter) {
const float zx2 = zx * zx;
const float zy2 = zy * zy;

if (zx2 + zy2 > 4.0f)
  break;

{
  const float new_zy = 2.0f * zx * zy + cy;
  const float new_zx = zx2 - zy2 + cx;
  zx = new_zx;
  zy = new_zy;
  iter++;
}

}

return iter;
}

int mandelbrot_masked_loop_prepare(struct vc4_runtime *rt,
struct mandelbrot_masked_loop_state *state,
uint32_t max_width,
uint32_t max_height) {
if (state == NULL)
return -1;
if (max_width > 32u || max_height > 32u)
return -1;

state->rt = rt;
state->max_width = max_width;
state->max_height = max_height;
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = MANDELBROT_MASKED_LOOP_ACTIVE_QPUS;
state->lanes = MANDELBROT_MASKED_LOOP_LANES;
return 0;
}

int mandelbrot_masked_loop_launch(struct mandelbrot_masked_loop_state *state,
uint32_t *output,
uint32_t width,
uint32_t height,
float min_x,
float min_y,
float step_x,
float step_y,
uint32_t max_iter) {
if (state == NULL || output == NULL)
return -1;
if (width > state->max_width || height > state->max_height)
return -1;

for (uint32_t y = 0; y < height; y++) {
const float cy = min_y + (float)y * step_y;
for (uint32_t x = 0; x < width; x++) {
const float cx = min_x + (float)x * step_x;
output[y * width + x] = mandelbrot_pixel_count(cx, cy, max_iter);
}
}

state->launch_count++;
return 0;
}

void mandelbrot_masked_loop_shutdown(struct mandelbrot_masked_loop_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
