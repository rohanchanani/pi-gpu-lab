#ifndef TMU_STRIDED_LOAD_LAUNCH_H
#define TMU_STRIDED_LOAD_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define TMU_STRIDED_LOAD_LANE_WIDTH 16u
#define TMU_STRIDED_LOAD_MAX_QPUS 12u
#define TMU_STRIDED_LOAD_GUARD_WORDS 32u
#define TMU_STRIDED_LOAD_SENTINEL (-12345.0f)

struct tmu_strided_load_state {
struct vc4_runtime *runtime;
void *opaque;
uint32_t prepared;
uint32_t max_input_words;
uint32_t max_n;
uint32_t padded_output_capacity_n;
uint32_t runtime_allocations;
uint32_t runtime_launches;
};

int tmu_strided_load_prepare(
struct vc4_runtime *rt,
struct tmu_strided_load_state *state,
uint32_t max_input_words,
uint32_t max_n);

int tmu_strided_load_launch(
struct tmu_strided_load_state *state,
const float *input,
float *out,
uint32_t n,
uint32_t offset,
uint32_t stride,
float scale,
float bias);

void tmu_strided_load_shutdown(struct tmu_strided_load_state *state);

uint32_t tmu_strided_load_runtime_allocations(
const struct tmu_strided_load_state *state);

uint32_t tmu_strided_load_runtime_launches(
const struct tmu_strided_load_state *state);

uint32_t tmu_strided_load_runtime_capacity_n(
const struct tmu_strided_load_state *state);

uint32_t tmu_strided_load_runtime_capacity_input_words(
const struct tmu_strided_load_state *state);

#endif

