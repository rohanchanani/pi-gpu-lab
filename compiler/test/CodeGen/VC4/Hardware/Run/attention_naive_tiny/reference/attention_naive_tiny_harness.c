#include "attention_naive_tiny_launch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define ATTENTION_NAIVE_TINY_CASES 6u
#define ATTENTION_NAIVE_TINY_MAX_Q_LEN 5u
#define ATTENTION_NAIVE_TINY_MAX_K_LEN 7u
#define ATTENTION_NAIVE_TINY_MAX_D 16u
#define ATTENTION_NAIVE_TINY_MAX_Q_WORDS (ATTENTION_NAIVE_TINY_MAX_Q_LEN * ATTENTION_NAIVE_TINY_MAX_D)
#define ATTENTION_NAIVE_TINY_MAX_K_WORDS (ATTENTION_NAIVE_TINY_MAX_K_LEN * ATTENTION_NAIVE_TINY_MAX_D)
#define ATTENTION_NAIVE_TINY_MAX_V_WORDS (ATTENTION_NAIVE_TINY_MAX_K_LEN * ATTENTION_NAIVE_TINY_MAX_D)
#define ATTENTION_NAIVE_TINY_MAX_OUT_WORDS (ATTENTION_NAIVE_TINY_MAX_Q_LEN * ATTENTION_NAIVE_TINY_MAX_D)
#define ATTENTION_NAIVE_TINY_GUARD_WORDS 64u
#define ATTENTION_NAIVE_TINY_SENTINEL (-12345.25f)
#define ATTENTION_NAIVE_TINY_TOLERANCE 0.05f

struct attention_naive_tiny_case {
uint32_t q_len;
uint32_t k_len;
uint32_t d;
float scale;
uint32_t equal_scores;
};

static const struct attention_naive_tiny_case attention_naive_tiny_cases[ATTENTION_NAIVE_TINY_CASES] = {
{0u, 4u, 8u, 0.353553f, 0u},
{2u, 0u, 8u, 0.353553f, 0u},
{1u, 1u, 1u, 1.0f, 0u},
{2u, 3u, 4u, 0.5f, 0u},
{3u, 4u, 8u, 0.353553f, 1u},
{5u, 7u, 16u, 0.25f, 0u},
};

static float q_values[ATTENTION_NAIVE_TINY_MAX_Q_WORDS];
static float k_values[ATTENTION_NAIVE_TINY_MAX_K_WORDS];
static float v_values[ATTENTION_NAIVE_TINY_MAX_V_WORDS];
static float out_values[ATTENTION_NAIVE_TINY_MAX_OUT_WORDS + ATTENTION_NAIVE_TINY_GUARD_WORDS];
static float expected_values[ATTENTION_NAIVE_TINY_MAX_OUT_WORDS];

static float abs_f32(float value) {
return value < 0.0f ? -value : value;
}

static float make_q_value(uint32_t q, uint32_t t, uint32_t case_id) {
const int raw = (int)((q * 11u + t * 7u + case_id * 5u + 3u) % 17u) - 8;
return (float)raw * 0.125f;
}

static float make_k_value(uint32_t k, uint32_t t, uint32_t case_id) {
const int raw = (int)((k * 13u + t * 3u + case_id * 9u + 1u) % 19u) - 9;
return (float)raw * 0.125f;
}

static float make_v_value(uint32_t k, uint32_t t, uint32_t case_id) {
const int raw = (int)((k * 5u + t * 11u + case_id * 7u + 4u) % 23u) - 11;
return (float)raw * 0.0625f;
}

static void fill_case(uint32_t case_id, uint32_t q_len, uint32_t k_len,
uint32_t d, float scale, uint32_t equal_scores) {
for (uint32_t i = 0; i < ATTENTION_NAIVE_TINY_MAX_Q_WORDS; i++)
q_values[i] = 0.0f;
for (uint32_t i = 0; i < ATTENTION_NAIVE_TINY_MAX_K_WORDS; i++)
k_values[i] = 0.0f;
for (uint32_t i = 0; i < ATTENTION_NAIVE_TINY_MAX_V_WORDS; i++)
v_values[i] = 0.0f;
for (uint32_t i = 0; i < ATTENTION_NAIVE_TINY_MAX_OUT_WORDS; i++)
expected_values[i] = 0.0f;
for (uint32_t i = 0; i < ATTENTION_NAIVE_TINY_MAX_OUT_WORDS + ATTENTION_NAIVE_TINY_GUARD_WORDS; i++)
out_values[i] = ATTENTION_NAIVE_TINY_SENTINEL;

for (uint32_t qi = 0; qi < q_len; qi++) {
for (uint32_t t = 0; t < d; t++) {
if (equal_scores)
q_values[qi * d + t] = 0.0f;
else
q_values[qi * d + t] = make_q_value(qi, t, case_id);
}
}

for (uint32_t ki = 0; ki < k_len; ki++) {
for (uint32_t t = 0; t < d; t++) {
if (equal_scores)
k_values[ki * d + t] = make_k_value(0u, t, case_id);
else
k_values[ki * d + t] = make_k_value(ki, t, case_id);
v_values[ki * d + t] = make_v_value(ki, t, case_id);
}
}

if (k_len == 0u) {
for (uint32_t qi = 0; qi < q_len; qi++) {
for (uint32_t t = 0; t < d; t++)
expected_values[qi * d + t] = 0.0f;
}
(void)scale;
return;
}

for (uint32_t qi = 0; qi < q_len; qi++) {
float scores[8];
float max_score = -3.4028234663852886e38f;
float sum_exp = 0.0f;

for (uint32_t ki = 0; ki < k_len; ki++) {
  float dot = 0.0f;
  for (uint32_t t = 0; t < d; t++)
    dot += q_values[qi * d + t] * k_values[ki * d + t];
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
    acc += p * v_values[ki * d + t];
  }
  expected_values[qi * d + t] = acc;
}

}
}

int main(void) {
struct attention_naive_tiny_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t launch_failures = 0u;
float global_max_abs_diff = 0.0f;
double checksum_accum = 0.0;
const clock_t start_clock = clock();

if (attention_naive_tiny_prepare(NULL, &state, ATTENTION_NAIVE_TINY_MAX_Q_LEN,
ATTENTION_NAIVE_TINY_MAX_K_LEN,
ATTENTION_NAIVE_TINY_MAX_D) != 0) {
printf("VC4_TEST_RESULT name=attention_naive_tiny status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 max_q_len=5 max_k_len=7 max_d=16 max_abs_diff=0.0 runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
return 1;
}

printf("ATTENTION_NAIVE_TINY_RUNTIME_SETUP max_q_len=%u max_k_len=%u max_d=%u allocations=%u\n",
state.max_q_len, state.max_k_len, state.max_d, state.allocation_count);

for (uint32_t case_id = 0; case_id < ATTENTION_NAIVE_TINY_CASES; case_id++) {
const uint32_t q_len = attention_naive_tiny_cases[case_id].q_len;
const uint32_t k_len = attention_naive_tiny_cases[case_id].k_len;
const uint32_t d = attention_naive_tiny_cases[case_id].d;
const float scale = attention_naive_tiny_cases[case_id].scale;
const uint32_t logical_outputs = q_len * d;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
float max_abs_diff = 0.0f;
double checksum = 0.0;

fill_case(case_id, q_len, k_len, d, scale,
          attention_naive_tiny_cases[case_id].equal_scores);

if (attention_naive_tiny_launch(&state, q_values, k_values, v_values,
                                out_values, q_len, k_len, d, scale) != 0) {
  launch_failures++;
  printf("ATTENTION_NAIVE_TINY_CASE case=%u q_len=%u k_len=%u d=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, q_len, k_len, d, state.launch_count,
         state.allocation_count);
  continue;
}

for (uint32_t i = 0; i < logical_outputs; i++) {
  const float diff = out_values[i] - expected_values[i];
  const float adiff = abs_f32(diff);
  checksum += (double)out_values[i] * 1024.0;

  if (adiff > max_abs_diff)
    max_abs_diff = adiff;

  if (adiff > ATTENTION_NAIVE_TINY_TOLERANCE) {
    if (mismatches < 8u) {
      const uint32_t qi = d == 0u ? 0u : i / d;
      const uint32_t lane = d == 0u ? 0u : i % d;
      printf("ERROR: case=%u q=%u lane=%u gpu=%f cpu=%f diff=%f\n",
             case_id, qi, lane, out_values[i], expected_values[i], diff);
    }
    mismatches++;
  }
}

for (uint32_t i = logical_outputs; i < ATTENTION_NAIVE_TINY_MAX_OUT_WORDS + ATTENTION_NAIVE_TINY_GUARD_WORDS; i++) {
  if (out_values[i] != ATTENTION_NAIVE_TINY_SENTINEL)
    sentinel_mismatches++;
}

if (max_abs_diff > global_max_abs_diff)
  global_max_abs_diff = max_abs_diff;

total_mismatches += mismatches;
total_sentinel_mismatches += sentinel_mismatches;
checksum_accum += checksum;

printf("ATTENTION_NAIVE_TINY_CASE case=%u q_len=%u k_len=%u d=%u mismatches=%u sentinel_mismatches=%u checksum=%.0f max_abs_diff=%.6f launches=%u allocations=%u\n",
       case_id, q_len, k_len, d, mismatches, sentinel_mismatches,
       checksum, max_abs_diff, state.launch_count, state.allocation_count);

}

attention_naive_tiny_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=attention_naive_tiny status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u max_q_len=%u max_k_len=%u max_d=%u checksum_accum=%.0f max_abs_diff=%.6f runtime_allocations=%u runtime_launches=%u elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= ATTENTION_NAIVE_TINY_TOLERANCE)
? "PASS"
: "FAIL",
ATTENTION_NAIVE_TINY_CASES, total_mismatches,
total_sentinel_mismatches, launch_failures, state.active_qpus,
state.lanes, state.max_q_len, state.max_k_len, state.max_d,
checksum_accum, global_max_abs_diff, state.allocation_count,
state.launch_count, elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= ATTENTION_NAIVE_TINY_TOLERANCE)
? 0
: 1;
}
