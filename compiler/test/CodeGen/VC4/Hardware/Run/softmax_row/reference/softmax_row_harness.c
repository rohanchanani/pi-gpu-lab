#include "softmax_row_launch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define SOFTMAX_ROW_CASES 7u
#define SOFTMAX_ROW_MAX_ROWS 13u
#define SOFTMAX_ROW_MAX_WIDTH 16u
#define SOFTMAX_ROW_MAX_WORDS (SOFTMAX_ROW_MAX_ROWS * SOFTMAX_ROW_MAX_WIDTH)
#define SOFTMAX_ROW_GUARD_WORDS 64u
#define SOFTMAX_ROW_SENTINEL (-12345.25f)

struct softmax_row_case {
uint32_t rows;
uint32_t width;
uint32_t equal_first_row;
};

static const struct softmax_row_case softmax_row_cases[SOFTMAX_ROW_CASES] = {
{0u, 8u, 0u},
{1u, 1u, 1u},
{2u, 4u, 0u},
{3u, 8u, 0u},
{5u, 15u, 1u},
{7u, 16u, 0u},
{13u, 9u, 0u},
};

static float input_values[SOFTMAX_ROW_MAX_WORDS];
static float output_values[SOFTMAX_ROW_MAX_WORDS + SOFTMAX_ROW_GUARD_WORDS];
static float expected_values[SOFTMAX_ROW_MAX_WORDS];

static float abs_f32(float value) {
return value < 0.0f ? -value : value;
}

static float make_input_value(uint32_t row, uint32_t lane, uint32_t case_id) {
const int raw = (int)((row * 13u + lane * 5u + case_id * 7u + 3u) % 33u) - 16;
return (float)raw * 0.125f;
}

static void fill_case(uint32_t case_id, uint32_t rows, uint32_t width,
uint32_t equal_first_row) {
for (uint32_t i = 0; i < SOFTMAX_ROW_MAX_WORDS; i++) {
input_values[i] = 0.0f;
expected_values[i] = 0.0f;
}

for (uint32_t i = 0; i < SOFTMAX_ROW_MAX_WORDS + SOFTMAX_ROW_GUARD_WORDS; i++)
output_values[i] = SOFTMAX_ROW_SENTINEL;

for (uint32_t row = 0; row < rows; row++) {
for (uint32_t lane = 0; lane < width; lane++) {
if (equal_first_row != 0u && row == 0u)
input_values[row * width + lane] = 0.375f;
else
input_values[row * width + lane] = make_input_value(row, lane, case_id);
}
}

for (uint32_t row = 0; row < rows; row++) {
const uint32_t base = row * width;

if (width == 0u)
  continue;

float max_value = input_values[base];
for (uint32_t lane = 1u; lane < width; lane++) {
  if (input_values[base + lane] > max_value)
    max_value = input_values[base + lane];
}

float sum = 0.0f;
for (uint32_t lane = 0; lane < width; lane++) {
  const float e = expf(input_values[base + lane] - max_value);
  expected_values[base + lane] = e;
  sum += e;
}

for (uint32_t lane = 0; lane < width; lane++)
  expected_values[base + lane] /= sum;

}
}

int main(void) {
struct softmax_row_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t total_row_sum_mismatches = 0u;
uint32_t launch_failures = 0u;
float global_max_abs_diff = 0.0f;
double checksum_accum = 0.0;
const clock_t start_clock = clock();

if (softmax_row_prepare(NULL, &state, SOFTMAX_ROW_MAX_ROWS,
SOFTMAX_ROW_MAX_WIDTH) != 0) {
printf("VC4_TEST_RESULT name=softmax_row status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 row_sum_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 max_rows=13 max_width=16 max_abs_diff=0.0 runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
return 1;
}

printf("SOFTMAX_ROW_RUNTIME_SETUP max_rows=%u max_width=%u allocations=%u\n",
state.max_rows, state.max_width, state.allocation_count);

for (uint32_t case_id = 0; case_id < SOFTMAX_ROW_CASES; case_id++) {
const uint32_t rows = softmax_row_cases[case_id].rows;
const uint32_t width = softmax_row_cases[case_id].width;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
uint32_t row_sum_mismatches = 0u;
float max_abs_diff = 0.0f;
double checksum = 0.0;
const uint32_t logical_words = rows * width;

fill_case(case_id, rows, width, softmax_row_cases[case_id].equal_first_row);

if (softmax_row_launch(&state, input_values, output_values, rows, width) != 0) {
  launch_failures++;
  printf("SOFTMAX_ROW_CASE case=%u rows=%u width=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, rows, width, state.launch_count, state.allocation_count);
  continue;
}

for (uint32_t i = 0; i < logical_words; i++) {
  const float diff = output_values[i] - expected_values[i];
  const float adiff = abs_f32(diff);
  checksum += (double)output_values[i] * 1024.0;

  if (adiff > max_abs_diff)
    max_abs_diff = adiff;

  if (adiff > 0.03f) {
    if (mismatches < 8u) {
      const uint32_t row = width == 0u ? 0u : i / width;
      const uint32_t lane = width == 0u ? 0u : i % width;
      printf("ERROR: case=%u row=%u lane=%u gpu=%f cpu=%f diff=%f\n",
             case_id, row, lane, output_values[i], expected_values[i], diff);
    }
    mismatches++;
  }
}

if (width != 0u) {
  for (uint32_t row = 0; row < rows; row++) {
    float row_sum = 0.0f;
    for (uint32_t lane = 0; lane < width; lane++)
      row_sum += output_values[row * width + lane];
    if (abs_f32(row_sum - 1.0f) > 0.03f) {
      if (row_sum_mismatches < 4u)
        printf("ERROR: case=%u row=%u softmax row_sum=%f\n", case_id, row, row_sum);
      row_sum_mismatches++;
    }
  }
}

for (uint32_t i = logical_words; i < SOFTMAX_ROW_MAX_WORDS + SOFTMAX_ROW_GUARD_WORDS; i++) {
  if (output_values[i] != SOFTMAX_ROW_SENTINEL)
    sentinel_mismatches++;
}

if (max_abs_diff > global_max_abs_diff)
  global_max_abs_diff = max_abs_diff;

total_mismatches += mismatches;
total_sentinel_mismatches += sentinel_mismatches;
total_row_sum_mismatches += row_sum_mismatches;
checksum_accum += checksum;

printf("SOFTMAX_ROW_CASE case=%u rows=%u width=%u mismatches=%u sentinel_mismatches=%u row_sum_mismatches=%u checksum=%.0f max_abs_diff=%.6f launches=%u allocations=%u\n",
       case_id, rows, width, mismatches, sentinel_mismatches,
       row_sum_mismatches, checksum, max_abs_diff, state.launch_count,
       state.allocation_count);

}

softmax_row_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=softmax_row status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u row_sum_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u max_rows=%u max_width=%u checksum_accum=%.0f max_abs_diff=%.6f runtime_allocations=%u runtime_launches=%u elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u &&
total_row_sum_mismatches == 0u && launch_failures == 0u &&
global_max_abs_diff <= 0.03f)
? "PASS"
: "FAIL",
SOFTMAX_ROW_CASES, total_mismatches, total_sentinel_mismatches,
total_row_sum_mismatches, launch_failures, state.active_qpus,
state.lanes, state.max_rows, state.max_width, checksum_accum,
global_max_abs_diff, state.allocation_count, state.launch_count,
elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
total_row_sum_mismatches == 0u && launch_failures == 0u &&
global_max_abs_diff <= 0.03f)
? 0
: 1;
}
