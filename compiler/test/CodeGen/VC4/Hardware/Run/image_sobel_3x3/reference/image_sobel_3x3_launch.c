#include "image_sobel_3x3_launch.h"

#include <stddef.h>

static uint32_t image_sobel_3x3_pixel(const uint32_t *input,
uint32_t width,
uint32_t height,
int x,
int y) {
if (x < 0)
x = 0;
if (y < 0)
y = 0;
if ((uint32_t)x >= width)
x = (int)width - 1;
if ((uint32_t)y >= height)
y = (int)height - 1;
return input[(uint32_t)y * width + (uint32_t)x] & 255u;
}

static int image_sobel_3x3_abs_i32(int value) {
return value < 0 ? -value : value;
}

static uint32_t image_sobel_3x3_eval_pixel(const uint32_t *input,
uint32_t width,
uint32_t height,
uint32_t x,
uint32_t y) {
const int ix = (int)x;
const int iy = (int)y;

const int p00 = (int)image_sobel_3x3_pixel(input, width, height, ix - 1, iy - 1);
const int p01 = (int)image_sobel_3x3_pixel(input, width, height, ix, iy - 1);
const int p02 = (int)image_sobel_3x3_pixel(input, width, height, ix + 1, iy - 1);
const int p10 = (int)image_sobel_3x3_pixel(input, width, height, ix - 1, iy);
const int p12 = (int)image_sobel_3x3_pixel(input, width, height, ix + 1, iy);
const int p20 = (int)image_sobel_3x3_pixel(input, width, height, ix - 1, iy + 1);
const int p21 = (int)image_sobel_3x3_pixel(input, width, height, ix, iy + 1);
const int p22 = (int)image_sobel_3x3_pixel(input, width, height, ix + 1, iy + 1);

const int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
const int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
int mag = image_sobel_3x3_abs_i32(gx) + image_sobel_3x3_abs_i32(gy);

if (mag > 255)
mag = 255;
return (uint32_t)mag;
}

int image_sobel_3x3_prepare(struct vc4_runtime *rt,
struct image_sobel_3x3_state *state,
uint32_t max_width,
uint32_t max_height) {
if (state == NULL || max_width == 0u || max_height == 0u)
return -1;

state->rt = rt;
state->max_width = max_width;
state->max_height = max_height;
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = IMAGE_SOBEL_3X3_ACTIVE_QPUS;
state->lanes = IMAGE_SOBEL_3X3_LANES;
return 0;
}

int image_sobel_3x3_launch(struct image_sobel_3x3_state *state,
const uint32_t *input,
uint32_t *output,
uint32_t width,
uint32_t height) {
if (state == NULL || input == NULL || output == NULL)
return -1;
if (width == 0u || height == 0u)
return -1;
if (width > state->max_width || height > state->max_height)
return -1;

for (uint32_t y = 0; y < height; y++) {
for (uint32_t x = 0; x < width; x++) {
output[y * width + x] =
image_sobel_3x3_eval_pixel(input, width, height, x, y);
}
}

state->launch_count++;
return 0;
}

void image_sobel_3x3_shutdown(struct image_sobel_3x3_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
