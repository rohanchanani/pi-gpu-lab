#ifndef WARP_REDUCE_SUM_LAUNCH_H
#define WARP_REDUCE_SUM_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define WARP_REDUCE_SUM_LANE_WIDTH 16u
#define WARP_REDUCE_SUM_MAX_QPUS 12u
#define WARP_REDUCE_SUM_GUARD_WORDS 32u
#define WARP_REDUCE_SUM_SENTINEL (-12345.0f)

struct warp_reduce_sum_state {
struct vc4_runtime *runtime;
void *opaque;
uint32_t prepared;
uint32_t max_n;
uint32_t padded_capacity_n;
uint32_t runtime_allocations;
uint32_t runtime_launches;
};

int warp_reduce_sum_prepare(
struct vc4_runtime *rt,
struct warp_reduce_sum_state *state,
uint32_t max_n);

int warp_reduce_sum_launch(
struct warp_reduce_sum_state *state,
const float *input,
float *out,
uint32_t n);

void warp_reduce_sum_shutdown(struct warp_reduce_sum_state *state);

uint32_t warp_reduce_sum_runtime_allocations(
const struct warp_reduce_sum_state *state);

uint32_t warp_reduce_sum_runtime_launches(
const struct warp_reduce_sum_state *state);

uint32_t warp_reduce_sum_runtime_capacity(
const struct warp_reduce_sum_state *state);

#endif

