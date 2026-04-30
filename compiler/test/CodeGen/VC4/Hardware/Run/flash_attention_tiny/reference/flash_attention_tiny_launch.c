#include "flash_attention_tiny_launch.h"

#include <math.h>
#include <stddef.h>

int flash_attention_tiny_prepare(struct vc4_runtime *rt,
struct flash_attention_tiny_state *state,
uint32_t max_q_len,
uint32_t max_k_len,
uint32_t max_d) {
if (state == NULL)
return -1;
if (max_q_len > 4u)
return -1;
if (max_k_len > 8u)
return -1;
if (max_d > FLASH_ATTENTION_TINY_LANES)
return -1;

state->rt = rt;
state->max_q_len = max_q_len;
state->max_k_len = max_k_len;
state->max_d = max_d;
state->allocation_count = 1u;
state->launch_count = 0u;
state->active_qpus = FLASH_ATTENTION_TINY_ACTIVE_QPUS;
state->lanes = FLASH_ATTENTION_TINY_LANES;
return 0;
}

int flash_attention_tiny_launch(struct flash_attention_tiny_state *state,
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
if (q_len > 4u || k_len > 8u || d > FLASH_ATTENTION_TINY_LANES)
return -1;

for (uint32_t qi = 0; qi < q_len; qi++) {
float acc[FLASH_ATTENTION_TINY_LANES];
float m = -3.4028234663852886e38f;
float l = 0.0f;

for (uint32_t t = 0; t < d; t++)
  acc[t] = 0.0f;

for (uint32_t ki = 0; ki < k_len; ki++) {
  float dot = 0.0f;
  float s;
  float m_new;
  float old_weight;
  float new_weight;
  float l_new;

  for (uint32_t t = 0; t < d; t++)
    dot += q[qi * d + t] * k[ki * d + t];

  s = dot * scale;
  m_new = (m > s) ? m : s;

  if (l == 0.0f) {
    old_weight = 0.0f;
  } else {
    old_weight = l * expf(m - m_new);
  }

  new_weight = expf(s - m_new);
  l_new = old_weight + new_weight;

  if (l_new != 0.0f) {
    const float old_scale = old_weight / l_new;
    const float new_scale = new_weight / l_new;
    for (uint32_t t = 0; t < d; t++)
      acc[t] = acc[t] * old_scale + v[ki * d + t] * new_scale;
  }

  m = m_new;
  l = l_new;
}

for (uint32_t t = 0; t < d; t++)
  out[qi * d + t] = acc[t];

}

state->launch_count++;
return 0;
}

void flash_attention_tiny_shutdown(struct flash_attention_tiny_state *state) {
if (state == NULL)
return;
state->rt = NULL;
}
