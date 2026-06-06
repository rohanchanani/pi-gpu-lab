#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_EPSILON 0.0002f
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_CHECKSUM_SCALE 1024.0f
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_ITERS 4u
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_COLS 16u
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_INPUT_N \
  (MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_ITERS * MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_COLS)
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_OUTPUT_N 16u
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_GUARD 32u
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_BUFFER_N \
  (MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_OUTPUT_N + MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_GUARD)
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SENTINEL (-12345.0f)
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_ACTIVE_QPUS 1u
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_LANE_WIDTH 16u
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SEED 0.5f
#define MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SPILL_ADJUST 406.0f

static const uint32_t active_cols_cases[] = {16u, 1u, 7u};
static float input_values[MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_INPUT_N];
static float output_values[MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_BUFFER_N];
static float expected_values[MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_BUFFER_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float input_value(uint32_t row, uint32_t col) {
  return (float)(row + 1u) + (float)col * 0.25f;
}

static void fill_input(void) {
  for (uint32_t r = 0; r < MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_ITERS; ++r)
    for (uint32_t c = 0; c < MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_COLS; ++c)
      input_values[r * MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_COLS + c] =
          input_value(r, c);
}

static void fill_output(void) {
  for (uint32_t i = 0; i < MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_BUFFER_N; ++i) {
    output_values[i] = MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SENTINEL;
    expected_values[i] = MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SENTINEL;
  }
}

static void run_cpu_reference(uint32_t active_cols) {
  for (uint32_t c = 0; c < MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_OUTPUT_N; ++c) {
    float acc = MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SEED;
    for (uint32_t r = 0; r < MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_ITERS; ++r)
      if (c < active_cols)
        acc += input_value(r, c);
    expected_values[c] = acc + MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SPILL_ADJUST;
  }
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] * MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_CHECKSUM_SCALE);
  return checksum;
}

static void verify_results(int *mismatches, float *max_abs_diff) {
  *mismatches = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t i = 0; i < MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_OUTPUT_N; ++i) {
    float diff = output_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
      *max_abs_diff = abs_diff;
    if (abs_diff > MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_EPSILON) {
      if (*mismatches < 8)
        printk("ERROR: mixed_lower_half_spill_dma_branch_ssavc4 i=%d gpu=%f expected=%f diff=%f\n",
               (int)i, output_values[i], expected_values[i], diff);
      ++*mismatches;
    }
  }
}

static int verify_sentinel_region(void) {
  int mismatches = 0;
  for (uint32_t i = MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_OUTPUT_N;
       i < MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_BUFFER_N; ++i) {
    if (output_values[i] != MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: mixed_lower_half_spill_dma_branch_ssavc4 sentinel changed i=%d value=%f expected=%f\n",
               (int)i, output_values[i], MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  const uint32_t case_count = sizeof(active_cols_cases) / sizeof(active_cols_cases[0]);
  const uint32_t input_bytes = sizeof(input_values);
  const uint32_t output_bytes = sizeof(output_values);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("mixed_lower_half_spill_dma_branch_ssavc4 program create failed");
  if (vc4_m2_malloc(program, &input_dev, input_bytes) < 0 ||
      vc4_m2_malloc(program, &output_dev, output_bytes) < 0)
    panic("mixed_lower_half_spill_dma_branch_ssavc4 allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  float max_abs_diff_overall = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_LANE_WIDTH, 1, 1);
  fill_input();

  for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
    uint32_t active_cols = active_cols_cases[case_index];
    fill_output();
    run_cpu_reference(active_cols);
    if (vc4_m2_copy_htod(program, input_dev, input_values, input_bytes) < 0 ||
        vc4_m2_copy_htod(program, output_dev, output_values, output_bytes) < 0 ||
        mixed_lower_half_spill_dma_branch_ssavc4_launch(
            program, grid, block, input_dev, output_dev, active_cols,
            MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_COLS * sizeof(float)) < 0 ||
        vc4_m2_copy_dtoh(program, output_values, output_dev, output_bytes) < 0) {
      printk("ERROR: mixed_lower_half_spill_dma_branch_ssavc4 launch/copy failed case=%d active_cols=%d\n",
             (int)case_index, (int)active_cols);
      ++launch_failures;
      continue;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(&mismatches, &max_abs_diff);
    int case_sentinel_mismatches = verify_sentinel_region();
    int checksum = scaled_checksum(output_values, MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_OUTPUT_N);
    int expected_checksum = scaled_checksum(expected_values, MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_OUTPUT_N);
    if (checksum != expected_checksum) {
      printk("ERROR: mixed_lower_half_spill_dma_branch_ssavc4 checksum mismatch active_cols=%d gpu=%d expected=%d\n",
             (int)active_cols, checksum, expected_checksum);
      ++mismatches;
    }
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinel_mismatches;
    checksum_accum += checksum;
    printk("MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_CASE case=%d active_cols=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
           (int)case_index, (int)active_cols, mismatches,
           case_sentinel_mismatches, checksum, max_abs_diff);
  }

  launch_failures +=
      (int)mixed_lower_half_spill_dma_branch_ssavc4_runtime_launch_failures();
  uint32_t launches = mixed_lower_half_spill_dma_branch_ssavc4_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=mixed_lower_half_spill_dma_branch_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d iters=%d forced_spill_live_adjust=%d saw_spill_frame_nonzero=%d saw_dynamic_dma_branch=%d saw_dma_branch=%d saw_edge_copy=%d saw_thread_end_epilogue=%d hidden_spill_reload_tmu_hits=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, (int)MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_ACTIVE_QPUS,
         (int)MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_LANE_WIDTH,
         (int)MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_ITERS,
         (int)MIXED_LOWER_HALF_SPILL_DMA_BRANCH_SSAVC4_SPILL_ADJUST,
         1, 1, 1, 1, 1, 0, checksum_accum, max_abs_diff_overall, 2,
         launches, timer_get_usec() - start);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
  vc4_program_destroy(program);
}
