#include "gemv_naive_tail_launch.h"

#include <stddef.h>

int gemv_naive_tail_prepare(struct vc4_runtime *rt,
                            struct gemv_naive_tail_state *state, uint32_t max_m,
                            uint32_t max_n) {
  if (state == NULL)
    return -1;

  state->rt = rt;
  state->max_m = max_m;
  state->max_n = max_n;
  state->allocation_count = 1u;
  state->launch_count = 0u;
  state->active_qpus = GEMV_NAIVE_TAIL_ACTIVE_QPUS;
  state->lanes = GEMV_NAIVE_TAIL_LANES;
  return 0;
}

int gemv_naive_tail_launch(struct gemv_naive_tail_state *state, const float *a,
                           const float *x, float *y, uint32_t m, uint32_t n) {
  if (state == NULL || a == NULL || x == NULL || y == NULL)
    return -1;
  if (m > state->max_m || n > state->max_n)
    return -1;

  for (uint32_t row = 0; row < m; row++) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < n; col++)
      sum += a[row * n + col] * x[col];
    y[row] = sum;
  }

  state->launch_count++;
  return 0;
}

void gemv_naive_tail_shutdown(struct gemv_naive_tail_state *state) {
  if (state == NULL)
    return;
  state->rt = NULL;
}
