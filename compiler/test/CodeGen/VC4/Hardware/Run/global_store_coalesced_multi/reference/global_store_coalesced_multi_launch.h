#ifndef GLOBAL_STORE_COALESCED_MULTI_LAUNCH_H
#define GLOBAL_STORE_COALESCED_MULTI_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define GLOBAL_STORE_COALESCED_MULTI_LANE_WIDTH 16u
#define GLOBAL_STORE_COALESCED_MULTI_MAX_QPUS 12u
#define GLOBAL_STORE_COALESCED_MULTI_GUARD_WORDS 32u
#define QPU_STORE_SENTINEL 0xdeadbeefu

struct global_store_coalesced_multi_state {
struct vc4_runtime *runtime;
void *opaque;
uint32_t prepared;
uint32_t max_n;
uint32_t padded_capacity_n;
uint32_t runtime_allocations;
uint32_t runtime_launches;
};

int global_store_coalesced_multi_prepare(
struct vc4_runtime *rt,
struct global_store_coalesced_multi_state *state,
uint32_t max_n);

int global_store_coalesced_multi_launch(
struct global_store_coalesced_multi_state *state,
uint32_t *out,
uint32_t n,
uint32_t case_id);

void global_store_coalesced_multi_shutdown(
struct global_store_coalesced_multi_state *state);

uint32_t global_store_coalesced_multi_runtime_allocations(
const struct global_store_coalesced_multi_state *state);

uint32_t global_store_coalesced_multi_runtime_launches(
const struct global_store_coalesced_multi_state *state);

uint32_t global_store_coalesced_multi_runtime_capacity(
const struct global_store_coalesced_multi_state *state);

#endif

