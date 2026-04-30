#include "attention_naive_tiny_launch.h"

#include <math.h>
#include <stddef.h>

int attention_naive_tiny_prepare(struct vc4_runtime *rt,
struct attention_naive_tiny_state *state,
uint32_t max_q_len,
uint32_t max_k_len,
uint32_t max_d) {
if (state == NULL)
return -1;
if (max_k_len > 8u)
return -1;
if (max_d > ATTENTION_NAIVE_TINY_LANES)
return -1;

state->rt = rt;
state->max_q_len = max_q_len;
state->max_k_len = max_k_len;
state->max_d = max_d;
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = ATTENTION_NAIVE_TINY_ACTIVE_QPUS;
state->lanes = ATTENTION_NAIVE_TINY_LANES;
return 0;
}

int attention_naive_tiny_launch(struct attention_naive_tiny_state *state,
const float *q,
const float *k,
const float *v,
float *out,
uint32_t q_len,
uint32_t k_len,
uint32_t d,
float scale) {
if (state == NULL || q == NULL || k == NULL || v == NULL || out == NULL)
return -1;
if (q_len > state->max_q_len || k_len > state->max_k_len || d > state->max_d)
return -1;
if (k_len > 8u || d > ATTENTION_NAIVE_TINY_LANES)
return -1;

if (k_len == 0u) {
for (uint32_t qi = 0; qi < q_len; qi++) {
for (uint32_t t = 0; t < d; t++)
out[qi * d + t] = 0.0f;
}
state->launch_count++;
return 0;
}

for (uint32_t qi = 0; qi < q_len; qi++) {
float scores[8];
float max_score = -3.4028234663852886e38f;
float sum_exp = 0.0f;

for (uint32_t ki = 0; ki < k_len; ki++) {
  float dot = 0.0f;
  for (uint32_t t = 0; t < d; t++)
    dot += q[qi * d + t] * k[ki * d + t];
  scores[ki] = dot * scale;
  if (scores[ki] > max_score)
    max_score = scores[ki];
}

for (uint32_t ki = 0; ki < k_len; ki++) {
  scores[ki] = expf(scores[ki] - max_score);
  sum_exp += scores[ki];
}

for (uint32_t t = 0; t < d; t++) {
  float acc = 0.0f;
  for (uint32_t ki = 0; ki < k_len; ki++) {
    const float p = scores[ki] / sum_exp;
    acc += p * v[ki * d + t];
  }
  out[qi * d + t] = acc;
}

}

state->launch_count++;
return 0;
}

void attention_naive_tiny_shutdown(struct attention_naive_tiny_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
