#include "attention_qk_naive_launch.h"

#include <stddef.h>

int attention_qk_naive_prepare(struct vc4_runtime *rt,
struct attention_qk_naive_state *state,
uint32_t max_q_len,
uint32_t max_k_len,
uint32_t max_d) {
if (state == NULL)
return -1;
if (max_k_len > ATTENTION_QK_NAIVE_LANES)
return -1;
if (max_d > ATTENTION_QK_NAIVE_LANES)
return -1;

state->rt = rt;
state->max_q_len = max_q_len;
state->max_k_len = max_k_len;
state->max_d = max_d;
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = ATTENTION_QK_NAIVE_ACTIVE_QPUS;
state->lanes = ATTENTION_QK_NAIVE_LANES;
return 0;
}

int attention_qk_naive_launch(struct attention_qk_naive_state *state,
const float *q,
const float *k,
float *scores,
uint32_t q_len,
uint32_t k_len,
uint32_t d,
float scale) {
if (state == NULL || q == NULL || k == NULL || scores == NULL)
return -1;
if (q_len > state->max_q_len || k_len > state->max_k_len || d > state->max_d)
return -1;
if (k_len > ATTENTION_QK_NAIVE_LANES || d > ATTENTION_QK_NAIVE_LANES)
return -1;

for (uint32_t qi = 0; qi < q_len; qi++) {
for (uint32_t ki = 0; ki < k_len; ki++) {
float acc = 0.0f;
for (uint32_t t = 0; t < d; t++)
acc += q[qi * d + t] * k[ki * d + t];
scores[qi * k_len + ki] = acc * scale;
}
}

state->launch_count++;
return 0;
}

void attention_qk_naive_shutdown(struct attention_qk_naive_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
