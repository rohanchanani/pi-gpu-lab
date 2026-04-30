#ifndef IMAGE_SOBEL_3X3_LAUNCH_H
#define IMAGE_SOBEL_3X3_LAUNCH_H

#include <stdint.h>

#define IMAGE_SOBEL_3X3_ACTIVE_QPUS 12u
#define IMAGE_SOBEL_3X3_LANES 16u

struct vc4_runtime;

struct image_sobel_3x3_state {
struct vc4_runtime *rt;
uint32_t max_width;
uint32_t max_height;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
};

int image_sobel_3x3_prepare(struct vc4_runtime *rt,
struct image_sobel_3x3_state *state,
uint32_t max_width,
uint32_t max_height);

int image_sobel_3x3_launch(struct image_sobel_3x3_state *state,
const uint32_t *input,
uint32_t *output,
uint32_t width,
uint32_t height);

void image_sobel_3x3_shutdown(struct image_sobel_3x3_state *state);

#endif
