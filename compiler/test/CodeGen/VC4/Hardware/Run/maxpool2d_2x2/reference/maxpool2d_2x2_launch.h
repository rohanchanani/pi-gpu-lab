#ifndef MAXPOOL2D_2X2_LAUNCH_H
#define MAXPOOL2D_2X2_LAUNCH_H

#include <stdint.h>

#define MAXPOOL2D_2X2_ACTIVE_QPUS 12u
#define MAXPOOL2D_2X2_LANES 16u

struct vc4_runtime;

struct maxpool2d_2x2_state {
struct vc4_runtime *rt;
uint32_t max_width;
uint32_t max_height;
uint32_t max_output_words;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
};

int maxpool2d_2x2_prepare(struct vc4_runtime *rt,
struct maxpool2d_2x2_state *state,
uint32_t max_width,
uint32_t max_height);

int maxpool2d_2x2_launch(struct maxpool2d_2x2_state *state,
const float *input,
float *output,
uint32_t width,
uint32_t height);

void maxpool2d_2x2_shutdown(struct maxpool2d_2x2_state *state);

#endif
