#ifndef BLOCK_REDUCE_SUM_LAUNCH_H
#define BLOCK_REDUCE_SUM_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define BLOCK_REDUCE_SUM_LANE_WIDTH 16u
#define BLOCK_REDUCE_SUM_MAX_QPUS 12u
#define BLOCK_REDUCE_SUM_MAX_WARPS_PER_BLOCK 12u
#define BLOCK_REDUCE_SUM_RESULT_LANES 16u
#define BLOCK_REDUCE_SUM_VPM_URSV_4K 16u
#define BLOCK_REDUCE_SUM_SENTINEL (-12345.0f)

struct block_reduce_sum_state {
struct vc4_runtime *runtime;
void *opaque;
uint32_t prepared;
uint32_t max_blocks;
uint32_t max_values_per_block;
uint32_t runtime_allocations;
uint32_t runtime_launches;
uint32_t runtime_tile_waves;
uint32_t runtime_timeouts;
uint32_t runtime_errstat_relevant_changed;
uint32_t runtime_vpmbase_readback;
uint32_t runtime_ident1;
uint32_t runtime_srqcs_after_last_wave;
};

int block_reduce_sum_prepare(
struct vc4_runtime *rt,
struct block_reduce_sum_state *state,
uint32_t max_blocks,
uint32_t max_values_per_block);

int block_reduce_sum_launch(
struct block_reduce_sum_state *state,
const float *input,
float *out,
uint32_t blocks,
uint32_t values_per_block,
uint32_t warps_per_block);

void block_reduce_sum_shutdown(struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_allocations(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_launches(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_tile_waves(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_timeouts(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_errstat_relevant_changed(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_capacity_blocks(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_capacity_values_per_block(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_vpmbase_readback(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_ident1(
const struct block_reduce_sum_state *state);

uint32_t block_reduce_sum_runtime_srqcs_after_last_wave(
const struct block_reduce_sum_state *state);

#endif

