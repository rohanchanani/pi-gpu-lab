#ifndef GEMV_NAIVE_TAIL_LAUNCH_H
#define GEMV_NAIVE_TAIL_LAUNCH_H

#include <stdint.h>

#define GEMV_NAIVE_TAIL_ACTIVE_QPUS 12u
#define GEMV_NAIVE_TAIL_LANES 16u

struct vc4_runtime;

struct gemv_naive_tail_state {
  struct vc4_runtime *rt;
  uint32_t max_m;
  uint32_t max_n;
  uint32_t allocation_count;
  uint32_t launch_count;
  uint32_t active_qpus;
  uint32_t lanes;
};

int gemv_naive_tail_prepare(struct vc4_runtime *rt,
                            struct gemv_naive_tail_state *state, uint32_t max_m,
                            uint32_t max_n);

int gemv_naive_tail_launch(struct gemv_naive_tail_state *state, const float *a,
                           const float *x, float *y, uint32_t m, uint32_t n);

void gemv_naive_tail_shutdown(struct gemv_naive_tail_state *state);

#endif
