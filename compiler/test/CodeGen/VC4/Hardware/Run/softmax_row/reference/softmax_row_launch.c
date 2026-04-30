#include "softmax_row_launch.h"

#include <math.h>
#include <stddef.h>

int softmax_row_prepare(struct vc4_runtime *rt,
struct softmax_row_state *state,
uint32_t max_rows,
uint32_t max_width) {
if (state == NULL)
return -1;
if (max_width > SOFTMAX_ROW_LANES)
return -1;

state->rt = rt;
state->max_rows = max_rows;
state->max_width = max_width;
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = SOFTMAX_ROW_ACTIVE_QPUS;
state->lanes = SOFTMAX_ROW_LANES;
return 0;
}

int softmax_row_launch(struct softmax_row_state *state,
const float *input,
float *output,
uint32_t rows,
uint32_t width) {
if (state == NULL || input == NULL || output == NULL)
return -1;
if (rows > state->max_rows || width > state->max_width)
return -1;
if (width > SOFTMAX_ROW_LANES)
return -1;

for (uint32_t row = 0; row < rows; row++) {
const uint32_t base = row * width;

if (width == 0u)
  continue;

float max_value = input[base];
for (uint32_t lane = 1u; lane < width; lane++) {
  if (input[base + lane] > max_value)
    max_value = input[base + lane];
}

float sum = 0.0f;
for (uint32_t lane = 0; lane < width; lane++) {
  const float e = expf(input[base + lane] - max_value);
  output[base + lane] = e;
  sum += e;
}

const float inv_sum = 1.0f / sum;
for (uint32_t lane = 0; lane < width; lane++)
  output[base + lane] *= inv_sum;

}

state->launch_count++;
return 0;
}

void softmax_row_shutdown(struct softmax_row_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
