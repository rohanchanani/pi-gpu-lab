#ifndef LAYERNORM_ROW_LAUNCH_H
#define LAYERNORM_ROW_LAUNCH_H

#include <stdint.h>

#define LAYERNORM_ROW_ACTIVE_QPUS 12u
#define LAYERNORM_ROW_LANES 16u

struct vc4_runtime;

struct layernorm_row_state {
struct vc4_runtime *rt;
uint32_t max_rows;
uint32_t max_width;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
};

int layernorm_row_prepare(struct vc4_runtime *rt,
struct layernorm_row_state *state,
uint32_t max_rows,
uint32_t max_width);

int layernorm_row_launch(struct layernorm_row_state *state,
const float *input,
float *output,
uint32_t rows,
uint32_t width,
float epsilon,
float gamma,
float beta);

void layernorm_row_shutdown(struct layernorm_row_state *state);

#endif
