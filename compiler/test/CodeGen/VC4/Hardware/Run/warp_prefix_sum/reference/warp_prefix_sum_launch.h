#ifndef WARP_PREFIX_SUM_LAUNCH_H
#define WARP_PREFIX_SUM_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define WARP_PREFIX_SUM_LANE_WIDTH 16u
#define WARP_PREFIX_SUM_MAX_QPUS 12u
#define WARP_PREFIX_SUM_GUARD_WORDS 32u
#define WARP_PREFIX_SUM_SENTINEL 0xdeadbeefu

struct warp_prefix_sum_state {
struct vc4_runtime *runtime;
void *opaque;
uint32_t prepared;
uint32_t max_n;
uint32_t padded_capacity_n;
uint32_t runtime_allocations;
uint32_t runtime_launches;
};

int warp_prefix_sum_prepare(
struct vc4_runtime *rt,
struct warp_prefix_sum_state *state,
uint32_t max_n);

int warp_prefix_sum_launch(
struct warp_prefix_sum_state *state,
const uint32_t *input,
uint32_t *out,
uint32_t n);

void warp_prefix_sum_shutdown(struct warp_prefix_sum_state *state);

uint32_t warp_prefix_sum_runtime_allocations(
const struct warp_prefix_sum_state *state);

uint32_t warp_prefix_sum_runtime_launches(
const struct warp_prefix_sum_state *state);

uint32_t warp_prefix_sum_runtime_capacity(
const struct warp_prefix_sum_state *state);

#endif

