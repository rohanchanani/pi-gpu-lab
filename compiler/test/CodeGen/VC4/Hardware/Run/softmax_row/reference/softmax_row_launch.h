#ifndef SOFTMAX_ROW_LAUNCH_H
#define SOFTMAX_ROW_LAUNCH_H

#include <stdint.h>

#define SOFTMAX_ROW_ACTIVE_QPUS 12u
#define SOFTMAX_ROW_LANES 16u

struct vc4_runtime;

struct softmax_row_state {
struct vc4_runtime *rt;
uint32_t max_rows;
uint32_t max_width;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
};

int softmax_row_prepare(struct vc4_runtime *rt,
struct softmax_row_state *state,
uint32_t max_rows,
uint32_t max_width);

int softmax_row_launch(struct softmax_row_state *state,
const float *input,
float *output,
uint32_t rows,
uint32_t width);

void softmax_row_shutdown(struct softmax_row_state *state);

#endif
