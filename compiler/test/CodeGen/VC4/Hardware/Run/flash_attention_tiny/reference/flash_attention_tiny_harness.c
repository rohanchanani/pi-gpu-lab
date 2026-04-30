#include "flash_attention_tiny_launch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define FLASH_ATTENTION_TINY_CASES 6u
#define FLASH_ATTENTION_TINY_MAX_Q_LEN 4u
#define FLASH_ATTENTION_TINY_MAX_K_LEN 8u
#define FLASH_ATTENTION_TINY_MAX_D 16u
#define FLASH_ATTENTION_TINY_MAX_Q_WORDS (FLASH_ATTENTION_TINY_MAX_Q_LEN * FLASH_ATTENTION_TINY_MAX_D)
#define FLASH_ATTENTION_TINY_MAX_K_WORDS (FLASH_ATTENTION_TINY_MAX_K_LEN * FLASH_ATTENTION_TINY_MAX_D)
#define FLASH_ATTENTION_TINY_MAX_V_WORDS (FLASH_ATTENTION_TINY_MAX_K_LEN * FLASH_ATTENTION_TINY_MAX_D)
#define FLASH_ATTENTION_TINY_MAX_OUT_WORDS (FLASH_ATTENTION_TINY_MAX_Q_LEN * FLASH_ATTENTION_TINY_MAX_D)
#define FLASH_ATTENTION_TINY_GUARD_WORDS 64u
#define FLASH_ATTENTION_TINY_SENTINEL (-23456.25f)
#define FLASH_ATTENTION_TINY_TOLERANCE 0.08f

struct flash_attention_tiny_case {
uint32_t q_len;
uint32_t k_len;
uint32_t d;
float scale;
uint32_t equal_scores;
uint32_t increasing_scores;
};

static const struct flash_attention_tiny_case flash_attention_tiny_cases[FLASH_ATTENTION_TINY_CASES] = {
{0u, 4u, 8u, 0.353553f, 0u, 0u},
{2u, 0u, 8u, 0.353553f, 0u, 0u},
{1u, 1u, 1u, 1.0f, 0u, 0u},
{2u, 3u, 4u, 0.5f, 1u, 0u},
{3u, 4u, 8u, 0.353553f, 0u, 1u},
{4u, 8u, 16u, 0.25f, 0u, 0u},
};

static float q_values[FLASH_ATTENTION_TINY_MAX_Q_WORDS];
static float k_values[FLASH_ATTENTION_TINY_MAX_K_WORDS];
static float v_values[FLASH_ATTENTION_TINY_MAX_V_WORDS];
static float out_values[FLASH_ATTENTION_TINY_MAX_OUT_WORDS + FLASH_ATTENTION_TINY_GUARD_WORDS];
static float expected_values[FLASH_ATTENTION_TINY_MAX_OUT_WORDS];

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

static void fill_case(uint32_t case_id, const struct flash_attention_tiny_case *tc) {
for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_Q_WORDS; i++)
q_values[i] = 0.0f;
for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_K_WORDS; i++)
k_values[i] = 0.0f;
for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_V_WORDS; i++)
v_values[i] = 0.0f;
for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_OUT_WORDS; i++)
expected_values[i] = 0.0f;
for (uint32_t i = 0; i < FLASH_ATTENTION_TINY_MAX_OUT_WORDS + FLASH_ATTENTION_TINY_GUARD_WORDS; i++)
out_values[i] = FLASH_ATTENTION_TINY_SENTINEL;

for (uint32_t qi = 0; qi < tc->q_len; qi++) {
for (uint32_t t = 0; t < tc->d; t++) {
if (tc->equal_scores)
q_values[qi * tc->d + t] = 0.0f;
else if (tc->increasing_scores)
q_values[qi * tc->d + t] = (t == 0u) ? (0.25f + (float)qi * 0.0625f) : 0.0f;
else
q_values[qi * tc->d + t] = make_q_value(qi, t, case_id);
}
}

for (uint32_t ki = 0; ki < tc->k_len; ki++) {
for (uint32_t t = 0; t < tc->d; t++) {
if (tc->equal_scores)
k_values[ki * tc->d + t] = make_k_value(0u, t, case_id);
else if (tc->increasing_scores)
k_values[ki * tc->d + t] = (t == 0u) ? (float)(ki + 1u) * 0.25f : 0.0f;
else
k_values[ki * tc->d + t] = make_k_value(ki, t, case_id);
v_values[ki * tc->d + t] = make_v_value(ki, t, case_id);
}
}
}

static void compute_expected(const struct flash_attention_tiny_case *tc) {
for (uint32_t qi = 0; qi < tc->q_len; qi++) {
float scores[8];
float max_score = -3.4028234663852886e38f;
float sum_exp = 0.0f;

for (uint32_t ki = 0; ki < tc->k_len; ki++) {
  float dot = 0.0f;
  for (uint32_t t = 0; t < tc->d; t++)
    dot += q_values[qi * tc->d + t] * k_values[ki * tc->d + t];
  scores[ki] = dot * tc->scale;
  if (scores[ki] > max_score)
    max_score = scores[ki];
}

for (uint32_t ki = 0; ki < tc->k_len; ki++) {
  scores[ki] = expf(scores[ki] - max_score);
  sum_exp += scores[ki];
}

for (uint32_t t = 0; t < tc->d; t++) {
  float acc = 0.0f;
  for (uint32_t ki = 0; ki < tc->k_len; ki++) {
    const float p = scores[ki] / sum_exp;
    acc += p * v_values[ki * tc->d + t];
  }
  expected_values[qi * tc->d + t] = acc;
}

}
}

int main(void) {
struct flash_attention_tiny_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t launch_failures = 0u;
float global_max_abs_diff = 0.0f;
double checksum_accum = 0.0;
const clock_t start_clock = clock();

if (flash_attention_tiny_prepare(NULL, &state, FLASH_ATTENTION_TINY_MAX_Q_LEN,
FLASH_ATTENTION_TINY_MAX_K_LEN,
FLASH_ATTENTION_TINY_MAX_D) != 0) {
printf("VC4_TEST_RESULT name=flash_attention_tiny status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 max_q_len=4 max_k_len=8 max_d=16 max_abs_diff=0.0 runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
return 1;
}

printf("FLASH_ATTENTION_TINY_RUNTIME_SETUP max_q_len=%u max_k_len=%u max_d=%u allocations=%u\n",
state.max_q_len, state.max_k_len, state.max_d, state.allocation_count);

for (uint32_t case_id = 0; case_id < FLASH_ATTENTION_TINY_CASES; case_id++) {
const struct flash_attention_tiny_case *tc = &flash_attention_tiny_cases[case_id];
const uint32_t logical_outputs = tc->q_len * tc->d;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
float max_abs_diff = 0.0f;
double checksum = 0.0;

fill_case(case_id, tc);
compute_expected(tc);

if (flash_attention_tiny_launch(&state, q_values, k_values, v_values,
                                out_values, tc->q_len, tc->k_len, tc->d,
                                tc->scale) != 0) {
  launch_failures++;
  printf("FLASH_ATTENTION_TINY_CASE case=%u q_len=%u k_len=%u d=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, tc->q_len, tc->k_len, tc->d, state.launch_count,
         state.allocation_count);
  continue;
}

for (uint32_t i = 0; i < logical_outputs; i++) {
  const float diff = out_values[i] - expected_values[i];
  const float adiff = abs_f32(diff);
  checksum += (double)out_values[i] * 1024.0;

  if (adiff > max_abs_diff)
    max_abs_diff = adiff;

  if (adiff > FLASH_ATTENTION_TINY_TOLERANCE) {
    if (mismatches < 8u) {
      const uint32_t qi = tc->d == 0u ? 0u : i / tc->d;
      const uint32_t lane = tc->d == 0u ? 0u : i % tc->d;
      printf("ERROR: case=%u q=%u lane=%u gpu=%f cpu=%f diff=%f\n",
             case_id, qi, lane, out_values[i], expected_values[i], diff);
    }
    mismatches++;
  }
}

for (uint32_t i = logical_outputs; i < FLASH_ATTENTION_TINY_MAX_OUT_WORDS + FLASH_ATTENTION_TINY_GUARD_WORDS; i++) {
  if (out_values[i] != FLASH_ATTENTION_TINY_SENTINEL)
    sentinel_mismatches++;
}

if (max_abs_diff > global_max_abs_diff)
  global_max_abs_diff = max_abs_diff;

total_mismatches += mismatches;
total_sentinel_mismatches += sentinel_mismatches;
checksum_accum += checksum;

printf("FLASH_ATTENTION_TINY_CASE case=%u q_len=%u k_len=%u d=%u mismatches=%u sentinel_mismatches=%u checksum=%.0f max_abs_diff=%.6f launches=%u allocations=%u\n",
       case_id, tc->q_len, tc->k_len, tc->d, mismatches,
       sentinel_mismatches, checksum, max_abs_diff, state.launch_count,
       state.allocation_count);

}

flash_attention_tiny_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=flash_attention_tiny status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u max_q_len=%u max_k_len=%u max_d=%u checksum_accum=%.0f max_abs_diff=%.6f runtime_allocations=%u runtime_launches=%u elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= FLASH_ATTENTION_TINY_TOLERANCE)
? "PASS"
: "FAIL",
FLASH_ATTENTION_TINY_CASES, total_mismatches,
total_sentinel_mismatches, launch_failures, state.active_qpus,
state.lanes, state.max_q_len, state.max_k_len, state.max_d,
checksum_accum, global_max_abs_diff, state.allocation_count,
state.launch_count, elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= FLASH_ATTENTION_TINY_TOLERANCE)
? 0
: 1;
}
