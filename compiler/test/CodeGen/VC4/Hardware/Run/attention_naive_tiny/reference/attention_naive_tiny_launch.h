#ifndef ATTENTION_NAIVE_TINY_LAUNCH_H
#define ATTENTION_NAIVE_TINY_LAUNCH_H

#include <stdint.h>

#define ATTENTION_NAIVE_TINY_ACTIVE_QPUS 12u
#define ATTENTION_NAIVE_TINY_LANES 16u

struct vc4_runtime;

struct attention_naive_tiny_state {
struct vc4_runtime *rt;
uint32_t max_q_len;
uint32_t max_k_len;
uint32_t max_d;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
};

int attention_naive_tiny_prepare(struct vc4_runtime *rt,
struct attention_naive_tiny_state *state,
uint32_t max_q_len,
uint32_t max_k_len,
uint32_t max_d);

int attention_naive_tiny_launch(struct attention_naive_tiny_state *state,
const float *q,
const float *k,
const float *v,
float *out,
uint32_t q_len,
uint32_t k_len,
uint32_t d,
float scale);

void attention_naive_tiny_shutdown(struct attention_naive_tiny_state *state);

#endif
