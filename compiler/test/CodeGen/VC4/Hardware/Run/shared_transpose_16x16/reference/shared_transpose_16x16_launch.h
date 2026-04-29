#ifndef SHARED_TRANSPOSE_16X16_LAUNCH_H
#define SHARED_TRANSPOSE_16X16_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define SHARED_TRANSPOSE_16X16_DIM 16u
#define SHARED_TRANSPOSE_16X16_WORDS 256u
#define SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK 4u
#define SHARED_TRANSPOSE_16X16_GUARD_WORDS 16u

struct shared_transpose_16x16_state
{
struct vc4_runtime *rt;
uint32_t prepared;
uint32_t active_qpus;
uint32_t lane_width;
uint32_t warps_per_block;
uint32_t runtime_allocations;
uint32_t runtime_launches;
uint32_t timeouts;
uint32_t errstat_relevant_changed;
uint32_t srqcs_after_last_launch;
uint32_t vpmbase_readback;
uint32_t last_sentinel_mismatches;
uint32_t total_sentinel_mismatches;
};

/*

Public semantic launcher API for shared_transpose_16x16.

Computes the exact transpose of one fixed row-major 16x16 uint32_t tile:

output[row][col] = input[col][row]

The public API exposes only the semantic input/output arrays, a runtime

handle supplied at prepare time, and a small non-hardware diagnostic state.

It does not expose qpu_id, logical warp id, raw uniform streams, VPM rows,

semaphore IDs, scheduler registers, code addresses, or bus addresses.
*/
int shared_transpose_16x16_prepare(
struct vc4_runtime *rt,
struct shared_transpose_16x16_state *state);

int shared_transpose_16x16_launch(
struct shared_transpose_16x16_state *state,
const uint32_t input[SHARED_TRANSPOSE_16X16_WORDS],
uint32_t output[SHARED_TRANSPOSE_16X16_WORDS]);

void shared_transpose_16x16_shutdown(
struct shared_transpose_16x16_state *state);

#endif

