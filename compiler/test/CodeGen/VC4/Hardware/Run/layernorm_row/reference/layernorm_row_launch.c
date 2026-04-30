#include "layernorm_row_launch.h"

#include <math.h>
#include <stddef.h>

int layernorm_row_prepare(struct vc4_runtime *rt,
struct layernorm_row_state *state,
uint32_t max_rows,
uint32_t max_width) {
if (state == NULL)
return -1;
if (max_width > LAYERNORM_ROW_LANES)
return -1;

state->rt = rt;
state->max_rows = max_rows;
state->max_width = max_width;
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = LAYERNORM_ROW_ACTIVE_QPUS;
state->lanes = LAYERNORM_ROW_LANES;
return 0;
}

int layernorm_row_launch(struct layernorm_row_state *state,
const float *input,
float *output,
uint32_t rows,
uint32_t width,
float epsilon,
float gamma,
float beta) {
if (state == NULL || input == NULL || output == NULL)
return -1;
if (rows > state->max_rows || width > state->max_width)
return -1;
if (width > LAYERNORM_ROW_LANES)
return -1;

for (uint32_t row = 0; row < rows; row++) {
const uint32_t base = row * width;
float mean = 0.0f;
float var = 0.0f;

if (width == 0u) {
  continue;
}

for (uint32_t lane = 0; lane < width; lane++)
  mean += input[base + lane];
mean /= (float)width;

for (uint32_t lane = 0; lane < width; lane++) {
  const float d = input[base + lane] - mean;
  var += d * d;
}
var /= (float)width;

const float inv_std = 1.0f / sqrtf(var + epsilon);
for (uint32_t lane = 0; lane < width; lane++)
  output[base + lane] = (input[base + lane] - mean) * inv_std * gamma + beta;

}

state->launch_count++;
return 0;
}

void layernorm_row_shutdown(struct layernorm_row_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
