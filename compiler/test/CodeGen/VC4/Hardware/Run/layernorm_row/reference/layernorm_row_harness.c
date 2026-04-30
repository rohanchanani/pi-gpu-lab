#include "layernorm_row_launch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define LAYERNORM_ROW_CASES 7u
#define LAYERNORM_ROW_MAX_ROWS 13u
#define LAYERNORM_ROW_MAX_WIDTH 16u
#define LAYERNORM_ROW_MAX_WORDS (LAYERNORM_ROW_MAX_ROWS * LAYERNORM_ROW_MAX_WIDTH)
#define LAYERNORM_ROW_GUARD_WORDS 64u
#define LAYERNORM_ROW_SENTINEL (-12345.25f)

struct layernorm_row_case {
uint32_t rows;
uint32_t width;
uint32_t constant_rows;
};

static const struct layernorm_row_case layernorm_row_cases[LAYERNORM_ROW_CASES] = {
{0u, 8u, 0u},
{1u, 1u, 1u},
{2u, 4u, 0u},
{3u, 8u, 0u},
{5u, 15u, 1u},
{7u, 16u, 0u},
{13u, 9u, 0u},
};

static float input_values[LAYERNORM_ROW_MAX_WORDS];
static float output_values[LAYERNORM_ROW_MAX_WORDS + LAYERNORM_ROW_GUARD_WORDS];
static float expected_values[LAYERNORM_ROW_MAX_WORDS];

static float abs_f32(float value) {
return value < 0.0f ? -value : value;
}

static float make_input_value(uint32_t row, uint32_t lane, uint32_t case_id) {
const int raw = (int)((row * 17u + lane * 7u + case_id * 5u + 11u) % 23u) - 11;
return (float)raw * 0.125f;
}

static void fill_case(uint32_t case_id, uint32_t rows, uint32_t width,
uint32_t constant_rows, float epsilon,
float gamma, float beta) {
for (uint32_t i = 0; i < LAYERNORM_ROW_MAX_WORDS; i++) {
input_values[i] = 0.0f;
expected_values[i] = 0.0f;
}

for (uint32_t i = 0; i < LAYERNORM_ROW_MAX_WORDS + LAYERNORM_ROW_GUARD_WORDS; i++)
output_values[i] = LAYERNORM_ROW_SENTINEL;

for (uint32_t row = 0; row < rows; row++) {
for (uint32_t lane = 0; lane < width; lane++) {
if (constant_rows != 0u && row == 0u)
input_values[row * width + lane] = 0.375f;
else
input_values[row * width + lane] = make_input_value(row, lane, case_id);
}
}

for (uint32_t row = 0; row < rows; row++) {
float mean = 0.0f;
float var = 0.0f;
const uint32_t base = row * width;

if (width == 0u)
  continue;

for (uint32_t lane = 0; lane < width; lane++)
  mean += input_values[base + lane];
mean /= (float)width;

for (uint32_t lane = 0; lane < width; lane++) {
  const float d = input_values[base + lane] - mean;
  var += d * d;
}
var /= (float)width;

const float inv_std = 1.0f / sqrtf(var + epsilon);
for (uint32_t lane = 0; lane < width; lane++)
  expected_values[base + lane] =
      (input_values[base + lane] - mean) * inv_std * gamma + beta;

}
}

int main(void) {
struct layernorm_row_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t launch_failures = 0u;
float global_max_abs_diff = 0.0f;
double checksum_accum = 0.0;
const float epsilon = 1.0e-3f;
const float gamma = 1.25f;
const float beta = -0.5f;
const clock_t start_clock = clock();

if (layernorm_row_prepare(NULL, &state, LAYERNORM_ROW_MAX_ROWS,
LAYERNORM_ROW_MAX_WIDTH) != 0) {
printf("VC4_TEST_RESULT name=layernorm_row status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 max_rows=13 max_width=16 max_abs_diff=0.0 runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
return 1;
}

printf("LAYERNORM_ROW_RUNTIME_SETUP max_rows=%u max_width=%u allocations=%u\n",
state.max_rows, state.max_width, state.allocation_count);

for (uint32_t case_id = 0; case_id < LAYERNORM_ROW_CASES; case_id++) {
const uint32_t rows = layernorm_row_cases[case_id].rows;
const uint32_t width = layernorm_row_cases[case_id].width;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
float max_abs_diff = 0.0f;
double checksum = 0.0;
const uint32_t logical_words = rows * width;

fill_case(case_id, rows, width, layernorm_row_cases[case_id].constant_rows,
          epsilon, gamma, beta);

if (layernorm_row_launch(&state, input_values, output_values, rows, width,
                         epsilon, gamma, beta) != 0) {
  launch_failures++;
  printf("LAYERNORM_ROW_CASE case=%u rows=%u width=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, rows, width, state.launch_count, state.allocation_count);
  continue;
}

for (uint32_t i = 0; i < logical_words; i++) {
  const float diff = output_values[i] - expected_values[i];
  const float adiff = abs_f32(diff);
  checksum += (double)output_values[i] * 1024.0;

  if (adiff > max_abs_diff)
    max_abs_diff = adiff;

  if (adiff > 0.02f) {
    if (mismatches < 8u) {
      const uint32_t row = width == 0u ? 0u : i / width;
      const uint32_t lane = width == 0u ? 0u : i % width;
      printf("ERROR: case=%u row=%u lane=%u gpu=%f cpu=%f diff=%f\n",
             case_id, row, lane, output_values[i], expected_values[i], diff);
    }
    mismatches++;
  }
}

for (uint32_t i = logical_words; i < LAYERNORM_ROW_MAX_WORDS + LAYERNORM_ROW_GUARD_WORDS; i++) {
  if (output_values[i] != LAYERNORM_ROW_SENTINEL)
    sentinel_mismatches++;
}

if (max_abs_diff > global_max_abs_diff)
  global_max_abs_diff = max_abs_diff;

total_mismatches += mismatches;
total_sentinel_mismatches += sentinel_mismatches;
checksum_accum += checksum;

printf("LAYERNORM_ROW_CASE case=%u rows=%u width=%u mismatches=%u sentinel_mismatches=%u checksum=%.0f max_abs_diff=%.6f launches=%u allocations=%u\n",
       case_id, rows, width, mismatches, sentinel_mismatches, checksum,
       max_abs_diff, state.launch_count, state.allocation_count);

}

layernorm_row_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=layernorm_row status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u max_rows=%u max_width=%u checksum_accum=%.0f max_abs_diff=%.6f runtime_allocations=%u runtime_launches=%u elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= 0.02f)
? "PASS"
: "FAIL",
LAYERNORM_ROW_CASES, total_mismatches, total_sentinel_mismatches,
launch_failures, state.active_qpus, state.lanes, state.max_rows,
state.max_width, checksum_accum, global_max_abs_diff,
state.allocation_count, state.launch_count, elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= 0.02f)
? 0
: 1;
}
