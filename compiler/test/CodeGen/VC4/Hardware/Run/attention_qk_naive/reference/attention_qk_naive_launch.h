#ifndef ATTENTION_QK_NAIVE_LAUNCH_H
#define ATTENTION_QK_NAIVE_LAUNCH_H

#include <stdint.h>

#define ATTENTION_QK_NAIVE_ACTIVE_QPUS 12u
#define ATTENTION_QK_NAIVE_LANES 16u

struct vc4_runtime;

struct attention_qk_naive_state {
struct vc4_runtime *rt;
uint32_t max_q_len;
uint32_t max_k_len;
uint32_t max_d;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
};

int attention_qk_naive_prepare(struct vc4_runtime *rt,
struct attention_qk_naive_state *state,
uint32_t max_q_len,
uint32_t max_k_len,
uint32_t max_d);

int attention_qk_naive_launch(struct attention_qk_naive_state *state,
const float *q,
const float *k,
float *scores,
uint32_t q_len,
uint32_t k_len,
uint32_t d,
float scale);

void attention_qk_naive_shutdown(struct attention_qk_naive_state *state);

#endif
