#ifndef FLASH_ATTENTION_TINY_LAUNCH_H
#define FLASH_ATTENTION_TINY_LAUNCH_H

#include <stdint.h>

#define FLASH_ATTENTION_TINY_ACTIVE_QPUS 12u
#define FLASH_ATTENTION_TINY_LANES 16u

struct vc4_runtime;

struct flash_attention_tiny_state {
struct vc4_runtime *rt;
uint32_t max_q_len;
uint32_t max_k_len;
uint32_t max_d;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
};

int flash_attention_tiny_prepare(struct vc4_runtime *rt,
struct flash_attention_tiny_state *state,
uint32_t max_q_len,
uint32_t max_k_len,
uint32_t max_d);

int flash_attention_tiny_launch(struct flash_attention_tiny_state *state,
const float *q,
const float *k,
const float *v,
float *out,
uint32_t q_len,
uint32_t k_len,
uint32_t d,
float scale);

void flash_attention_tiny_shutdown(struct flash_attention_tiny_state *state);

#endif
