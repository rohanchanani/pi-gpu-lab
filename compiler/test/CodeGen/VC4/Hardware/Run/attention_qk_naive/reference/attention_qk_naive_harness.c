#include "attention_qk_naive_launch.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define ATTENTION_QK_NAIVE_CASES 7u
#define ATTENTION_QK_NAIVE_MAX_Q_LEN 7u
#define ATTENTION_QK_NAIVE_MAX_K_LEN 16u
#define ATTENTION_QK_NAIVE_MAX_D 16u
#define ATTENTION_QK_NAIVE_MAX_Q_WORDS (ATTENTION_QK_NAIVE_MAX_Q_LEN * ATTENTION_QK_NAIVE_MAX_D)
#define ATTENTION_QK_NAIVE_MAX_K_WORDS (ATTENTION_QK_NAIVE_MAX_K_LEN * ATTENTION_QK_NAIVE_MAX_D)
#define ATTENTION_QK_NAIVE_MAX_SCORE_WORDS (ATTENTION_QK_NAIVE_MAX_Q_LEN * ATTENTION_QK_NAIVE_MAX_K_LEN)
#define ATTENTION_QK_NAIVE_GUARD_WORDS 64u
#define ATTENTION_QK_NAIVE_SENTINEL (-9876.5f)

struct attention_qk_naive_case {
uint32_t q_len;
uint32_t k_len;
uint32_t d;
float scale;
};

static const struct attention_qk_naive_case attention_qk_naive_cases[ATTENTION_QK_NAIVE_CASES] = {
{0u, 4u, 5u, 1.0f},
{3u, 0u, 5u, 1.0f},
{1u, 1u, 1u, 1.0f},
{2u, 3u, 4u, 0.5f},
{4u, 7u, 8u, 0.3535533905932738f},
{5u, 15u, 13u, 0.2773500981126146f},
{7u, 16u, 16u, 0.25f},
};

static float q_values[ATTENTION_QK_NAIVE_MAX_Q_WORDS];
static float k_values[ATTENTION_QK_NAIVE_MAX_K_WORDS];
static float score_values[ATTENTION_QK_NAIVE_MAX_SCORE_WORDS + ATTENTION_QK_NAIVE_GUARD_WORDS];
static float expected_values[ATTENTION_QK_NAIVE_MAX_SCORE_WORDS];

static float abs_f32(float value) {
return value < 0.0f ? -value : value;
}

static float make_q_value(uint32_t q, uint32_t t, uint32_t case_id) {
const int raw = (int)((q * 11u + t * 7u + case_id * 5u + 3u) % 19u) - 9;
return (float)raw * 0.125f;
}

static float make_k_value(uint32_t k, uint32_t t, uint32_t case_id) {
const int raw = (int)((k * 13u + t * 3u + case_id * 9u + 1u) % 23u) - 11;
return (float)raw * 0.125f;
}

static void fill_case(uint32_t case_id, uint32_t q_len, uint32_t k_len,
uint32_t d, float scale) {
for (uint32_t i = 0; i < ATTENTION_QK_NAIVE_MAX_Q_WORDS; i++)
q_values[i] = 0.0f;
for (uint32_t i = 0; i < ATTENTION_QK_NAIVE_MAX_K_WORDS; i++)
k_values[i] = 0.0f;
for (uint32_t i = 0; i < ATTENTION_QK_NAIVE_MAX_SCORE_WORDS; i++)
expected_values[i] = 0.0f;
for (uint32_t i = 0; i < ATTENTION_QK_NAIVE_MAX_SCORE_WORDS + ATTENTION_QK_NAIVE_GUARD_WORDS; i++)
score_values[i] = ATTENTION_QK_NAIVE_SENTINEL;

for (uint32_t qi = 0; qi < q_len; qi++) {
for (uint32_t t = 0; t < d; t++)
q_values[qi * d + t] = make_q_value(qi, t, case_id);
}

for (uint32_t ki = 0; ki < k_len; ki++) {
for (uint32_t t = 0; t < d; t++)
k_values[ki * d + t] = make_k_value(ki, t, case_id);
}

for (uint32_t qi = 0; qi < q_len; qi++) {
for (uint32_t ki = 0; ki < k_len; ki++) {
float acc = 0.0f;
for (uint32_t t = 0; t < d; t++)
acc += q_values[qi * d + t] * k_values[ki * d + t];
expected_values[qi * k_len + ki] = acc * scale;
}
}
}

int main(void) {
struct attention_qk_naive_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t launch_failures = 0u;
float global_max_abs_diff = 0.0f;
double checksum_accum = 0.0;
const clock_t start_clock = clock();

if (attention_qk_naive_prepare(NULL, &state, ATTENTION_QK_NAIVE_MAX_Q_LEN,
ATTENTION_QK_NAIVE_MAX_K_LEN,
ATTENTION_QK_NAIVE_MAX_D) != 0) {
printf("VC4_TEST_RESULT name=attention_qk_naive status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 max_q_len=7 max_k_len=16 max_d=16 max_abs_diff=0.0 runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
return 1;
}

printf("ATTENTION_QK_NAIVE_RUNTIME_SETUP max_q_len=%u max_k_len=%u max_d=%u allocations=%u\n",
state.max_q_len, state.max_k_len, state.max_d, state.allocation_count);

for (uint32_t case_id = 0; case_id < ATTENTION_QK_NAIVE_CASES; case_id++) {
const uint32_t q_len = attention_qk_naive_cases[case_id].q_len;
const uint32_t k_len = attention_qk_naive_cases[case_id].k_len;
const uint32_t d = attention_qk_naive_cases[case_id].d;
const float scale = attention_qk_naive_cases[case_id].scale;
const uint32_t logical_scores = q_len * k_len;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
float max_abs_diff = 0.0f;
double checksum = 0.0;

fill_case(case_id, q_len, k_len, d, scale);

if (attention_qk_naive_launch(&state, q_values, k_values, score_values,
                              q_len, k_len, d, scale) != 0) {
  launch_failures++;
  printf("ATTENTION_QK_NAIVE_CASE case=%u q_len=%u k_len=%u d=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, q_len, k_len, d, state.launch_count,
         state.allocation_count);
  continue;
}

for (uint32_t i = 0; i < logical_scores; i++) {
  const float diff = score_values[i] - expected_values[i];
  const float adiff = abs_f32(diff);
  checksum += (double)score_values[i] * 1024.0;

  if (adiff > max_abs_diff)
    max_abs_diff = adiff;

  if (adiff > 0.001f) {
    if (mismatches < 8u) {
      const uint32_t qi = k_len == 0u ? 0u : i / k_len;
      const uint32_t ki = k_len == 0u ? 0u : i % k_len;
      printf("ERROR: case=%u q=%u k=%u gpu=%f cpu=%f diff=%f\n",
             case_id, qi, ki, score_values[i], expected_values[i], diff);
    }
    mismatches++;
  }
}

for (uint32_t i = logical_scores; i < ATTENTION_QK_NAIVE_MAX_SCORE_WORDS + ATTENTION_QK_NAIVE_GUARD_WORDS; i++) {
  if (score_values[i] != ATTENTION_QK_NAIVE_SENTINEL)
    sentinel_mismatches++;
}

if (max_abs_diff > global_max_abs_diff)
  global_max_abs_diff = max_abs_diff;

total_mismatches += mismatches;
total_sentinel_mismatches += sentinel_mismatches;
checksum_accum += checksum;

printf("ATTENTION_QK_NAIVE_CASE case=%u q_len=%u k_len=%u d=%u mismatches=%u sentinel_mismatches=%u checksum=%.0f max_abs_diff=%.6f launches=%u allocations=%u\n",
       case_id, q_len, k_len, d, mismatches, sentinel_mismatches,
       checksum, max_abs_diff, state.launch_count, state.allocation_count);

}

attention_qk_naive_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=attention_qk_naive status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u max_q_len=%u max_k_len=%u max_d=%u checksum_accum=%.0f max_abs_diff=%.6f runtime_allocations=%u runtime_launches=%u elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= 0.001f)
? "PASS"
: "FAIL",
ATTENTION_QK_NAIVE_CASES, total_mismatches,
total_sentinel_mismatches, launch_failures, state.active_qpus,
state.lanes, state.max_q_len, state.max_k_len, state.max_d,
checksum_accum, global_max_abs_diff, state.allocation_count,
state.launch_count, elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= 0.001f)
? 0
: 1;
}
