#include "gemv_naive_tail_launch.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define GEMV_NAIVE_TAIL_CASES 9u
#define GEMV_NAIVE_TAIL_MAX_M 25u
#define GEMV_NAIVE_TAIL_MAX_N 33u
#define GEMV_NAIVE_TAIL_MAX_A_WORDS                                            \
  (GEMV_NAIVE_TAIL_MAX_M * GEMV_NAIVE_TAIL_MAX_N)
#define GEMV_NAIVE_TAIL_GUARD_WORDS 64u
#define GEMV_NAIVE_TAIL_SENTINEL (-9876.5f)

struct gemv_naive_tail_case {
  uint32_t m;
  uint32_t n;
};

static const struct gemv_naive_tail_case
    gemv_naive_tail_cases[GEMV_NAIVE_TAIL_CASES] = {
        {0u, 7u},  {4u, 0u},   {1u, 1u},   {3u, 5u},   {5u, 16u},
        {7u, 17u}, {12u, 31u}, {19u, 13u}, {25u, 33u},
};

static float a_values[GEMV_NAIVE_TAIL_MAX_A_WORDS];
static float x_values[GEMV_NAIVE_TAIL_MAX_N];
static float y_values[GEMV_NAIVE_TAIL_MAX_M + GEMV_NAIVE_TAIL_GUARD_WORDS];
static float expected_values[GEMV_NAIVE_TAIL_MAX_M];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }

static float make_a_value(uint32_t row, uint32_t col, uint32_t case_id) {
  const int raw = (int)((row * 11u + col * 7u + case_id * 5u + 3u) % 17u) - 8;
  return (float)raw * 0.125f;
}

static float make_x_value(uint32_t col, uint32_t case_id) {
  const int raw = (int)((col * 13u + case_id * 3u + 1u) % 19u) - 9;
  return (float)raw * 0.25f;
}

static void fill_case(uint32_t case_id, uint32_t m, uint32_t n) {
  for (uint32_t i = 0; i < GEMV_NAIVE_TAIL_MAX_A_WORDS; i++)
    a_values[i] = 0.0f;

  for (uint32_t i = 0; i < GEMV_NAIVE_TAIL_MAX_N; i++)
    x_values[i] = make_x_value(i, case_id);

  for (uint32_t i = 0; i < GEMV_NAIVE_TAIL_MAX_M; i++)
    expected_values[i] = 0.0f;

  for (uint32_t i = 0; i < GEMV_NAIVE_TAIL_MAX_M + GEMV_NAIVE_TAIL_GUARD_WORDS;
       i++)
    y_values[i] = GEMV_NAIVE_TAIL_SENTINEL;

  for (uint32_t row = 0; row < m; row++) {
    for (uint32_t col = 0; col < n; col++)
      a_values[row * n + col] = make_a_value(row, col, case_id);
  }

  for (uint32_t row = 0; row < m; row++) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < n; col++)
      sum += a_values[row * n + col] * x_values[col];
    expected_values[row] = sum;
  }
}

int main(void) {
  struct gemv_naive_tail_state state;
  uint32_t total_mismatches = 0u;
  uint32_t total_sentinel_mismatches = 0u;
  uint32_t launch_failures = 0u;
  float global_max_abs_diff = 0.0f;
  double checksum_accum = 0.0;
  const clock_t start_clock = clock();

  if (gemv_naive_tail_prepare(NULL, &state, GEMV_NAIVE_TAIL_MAX_M,
                              GEMV_NAIVE_TAIL_MAX_N) != 0) {
    printf("VC4_TEST_RESULT name=gemv_naive_tail status=FAIL cases=0 "
           "total_mismatches=0 sentinel_mismatches=0 launch_failures=1 "
           "active_qpus=12 lanes=16 max_m=25 max_n=33 max_abs_diff=0.0 "
           "runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
    return 1;
  }

  printf("GEMV_NAIVE_TAIL_RUNTIME_SETUP max_m=%u max_n=%u allocations=%u\n",
         state.max_m, state.max_n, state.allocation_count);

  for (uint32_t case_id = 0; case_id < GEMV_NAIVE_TAIL_CASES; case_id++) {
    const uint32_t m = gemv_naive_tail_cases[case_id].m;
    const uint32_t n = gemv_naive_tail_cases[case_id].n;
    uint32_t mismatches = 0u;
    uint32_t sentinel_mismatches = 0u;
    float max_abs_diff = 0.0f;
    double checksum = 0.0;

    fill_case(case_id, m, n);

    if (gemv_naive_tail_launch(&state, a_values, x_values, y_values, m, n) !=
        0) {
      launch_failures++;
      printf("GEMV_NAIVE_TAIL_CASE case=%u m=%u n=%u launch=FAIL launches=%u "
             "allocations=%u\n",
             case_id, m, n, state.launch_count, state.allocation_count);
      continue;
    }

    for (uint32_t row = 0; row < m; row++) {
      const float diff = y_values[row] - expected_values[row];
      const float adiff = abs_f32(diff);
      checksum += (double)y_values[row] * 1024.0;

      if (adiff > max_abs_diff)
        max_abs_diff = adiff;

      if (adiff > 0.001f) {
        if (mismatches < 8u) {
          printf("ERROR: case=%u row=%u gpu=%f cpu=%f diff=%f\n", case_id, row,
                 y_values[row], expected_values[row], diff);
        }
        mismatches++;
      }
    }

    for (uint32_t i = m;
         i < GEMV_NAIVE_TAIL_MAX_M + GEMV_NAIVE_TAIL_GUARD_WORDS; i++) {
      if (y_values[i] != GEMV_NAIVE_TAIL_SENTINEL)
        sentinel_mismatches++;
    }

    if (max_abs_diff > global_max_abs_diff)
      global_max_abs_diff = max_abs_diff;

    total_mismatches += mismatches;
    total_sentinel_mismatches += sentinel_mismatches;
    checksum_accum += checksum;

    printf("GEMV_NAIVE_TAIL_CASE case=%u m=%u n=%u mismatches=%u "
           "sentinel_mismatches=%u checksum=%.0f max_abs_diff=%.6f launches=%u "
           "allocations=%u\n",
           case_id, m, n, mismatches, sentinel_mismatches, checksum,
           max_abs_diff, state.launch_count, state.allocation_count);
  }

  gemv_naive_tail_shutdown(&state);

  clock_t end_clock = clock();
  unsigned long long elapsed_usec = 1ull;
  if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 &&
      end_clock >= start_clock) {
    elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
                   (unsigned long long)CLOCKS_PER_SEC;
    if (elapsed_usec == 0ull)
      elapsed_usec = 1ull;
  }

  printf("VC4_TEST_RESULT name=gemv_naive_tail status=%s cases=%u "
         "total_mismatches=%u sentinel_mismatches=%u launch_failures=%u "
         "active_qpus=%u lanes=%u max_m=%u max_n=%u checksum_accum=%.0f "
         "max_abs_diff=%.6f runtime_allocations=%u runtime_launches=%u "
         "elapsed_usec=%llu\n",
         (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
          launch_failures == 0u && global_max_abs_diff <= 0.001f)
             ? "PASS"
             : "FAIL",
         GEMV_NAIVE_TAIL_CASES, total_mismatches, total_sentinel_mismatches,
         launch_failures, state.active_qpus, state.lanes, state.max_m,
         state.max_n, checksum_accum, global_max_abs_diff,
         state.allocation_count, state.launch_count, elapsed_usec);

  return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
          launch_failures == 0u && global_max_abs_diff <= 0.001f)
             ? 0
             : 1;
}
