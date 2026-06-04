#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 4u
#define COLS 16u
#define MAX_PITCH_WORDS 31u
#define MAX_STRIDE_WORDS 31u
#define INPUT_WORDS (((ROWS - 1u) * MAX_PITCH_WORDS) + COLS)
#define FULL_OUTPUT_WORDS (ROWS * COLS)
#define SPARSE_OUTPUT_WORDS (((ROWS - 1u) * MAX_STRIDE_WORDS) + COLS)
#define GUARD_WORDS 16u
#define SENTINEL 0x8accee31u

static uint32_t input_values[INPUT_WORDS + GUARD_WORDS];
static uint32_t full_output[FULL_OUTPUT_WORDS + GUARD_WORDS];
static uint32_t sparse_output[SPARSE_OUTPUT_WORDS + GUARD_WORDS];
static const uint32_t active_cols_cases[] = {0u, 1u, 7u, 16u, 19u};
static const uint32_t pitch_bytes_cases[] = {64u, 76u, 124u};
static const uint32_t stride_bytes_cases[] = {64u, 76u, 124u};

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x5a000000u | (row << 8) | col;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t full_dev = 0;
  vc4_deviceptr_t sparse_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t zero_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_active_cols_multirow_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &full_dev, sizeof(full_output)) < 0 ||
      vc4Malloc(program, &sparse_dev, sizeof(sparse_output)) < 0)
    panic("dynamic_active_cols_multirow_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t p = 0;
       p < sizeof(pitch_bytes_cases) / sizeof(pitch_bytes_cases[0]); ++p) {
    uint32_t pitch_bytes = pitch_bytes_cases[p];
    uint32_t pitch_words = pitch_bytes / 4u;
    for (uint32_t s = 0;
         s < sizeof(stride_bytes_cases) / sizeof(stride_bytes_cases[0]); ++s) {
      uint32_t stride_bytes = stride_bytes_cases[s];
      uint32_t stride_words = stride_bytes / 4u;
      for (uint32_t c = 0;
           c < sizeof(active_cols_cases) / sizeof(active_cols_cases[0]); ++c) {
        uint32_t active_cols = active_cols_cases[c];
        uint32_t expected_cols = active_cols < COLS ? active_cols : COLS;
        for (uint32_t i = 0; i < INPUT_WORDS + GUARD_WORDS; ++i)
          input_values[i] = SENTINEL;
        for (uint32_t i = 0; i < FULL_OUTPUT_WORDS + GUARD_WORDS; ++i)
          full_output[i] = SENTINEL;
        for (uint32_t i = 0; i < SPARSE_OUTPUT_WORDS + GUARD_WORDS; ++i)
          sparse_output[i] = SENTINEL;
        for (uint32_t row = 0; row < ROWS; ++row)
          for (uint32_t col = 0; col < COLS; ++col)
            input_values[row * pitch_words + col] = pattern(row, col);

        if (vc4MemcpyHtoD(program, input_dev, input_values,
                          sizeof(input_values)) < 0 ||
            vc4MemcpyHtoD(program, full_dev, full_output,
                          sizeof(full_output)) < 0 ||
            vc4MemcpyHtoD(program, sparse_dev, sparse_output,
                          sizeof(sparse_output)) < 0 ||
            dynamic_active_cols_multirow_ssavc4_launch(
                program, grid, block, input_dev, full_dev, sparse_dev,
                active_cols, pitch_bytes, stride_bytes) < 0 ||
            vc4MemcpyDtoH(program, full_output, full_dev,
                          sizeof(full_output)) < 0 ||
            vc4MemcpyDtoH(program, sparse_output, sparse_dev,
                          sizeof(sparse_output)) < 0) {
          printk("ERROR: dynamic_active_cols_multirow launch/copy failed pitch_bytes=%d stride_bytes=%d active_cols=%d\n",
                 (int)pitch_bytes, (int)stride_bytes, (int)active_cols);
          ++launch_failures;
        }

        for (uint32_t row = 0; row < ROWS; ++row) {
          for (uint32_t col = 0; col < COLS; ++col) {
            uint32_t index = row * COLS + col;
            uint32_t expected =
                col < expected_cols ? pattern(row, col) : 0u;
            uint32_t actual = full_output[index];
            checksum += actual;
            if (actual != expected) {
              if (expected == 0u) {
                if (zero_mismatches < 8)
                  printk("ERROR: dynamic_active_cols_multirow full_zero pitch_bytes=%d active_cols=%d row=%d col=%d actual=%x\n",
                         (int)pitch_bytes, (int)active_cols, (int)row,
                         (int)col, actual);
                ++zero_mismatches;
              } else {
                if (total_mismatches < 8)
                  printk("ERROR: dynamic_active_cols_multirow full pitch_bytes=%d active_cols=%d row=%d col=%d actual=%x expected=%x\n",
                         (int)pitch_bytes, (int)active_cols, (int)row,
                         (int)col, actual, expected);
                ++total_mismatches;
              }
            }
          }
        }

        for (uint32_t i = 0; i < SPARSE_OUTPUT_WORDS + GUARD_WORDS; ++i) {
          uint32_t expected = SENTINEL;
          for (uint32_t row = 0; row < ROWS; ++row) {
            uint32_t row_base = row * stride_words;
            if (i >= row_base && i < row_base + expected_cols) {
              expected = pattern(row, i - row_base);
              break;
            }
          }
          uint32_t actual = sparse_output[i];
          if (expected != SENTINEL)
            checksum += actual;
          if (actual != expected) {
            if (expected == SENTINEL) {
              if (sentinel_mismatches < 8)
                printk("ERROR: dynamic_active_cols_multirow sparse_sentinel stride_bytes=%d active_cols=%d index=%d actual=%x\n",
                       (int)stride_bytes, (int)active_cols, (int)i, actual);
              ++sentinel_mismatches;
            } else {
              if (total_mismatches < 8)
                printk("ERROR: dynamic_active_cols_multirow sparse stride_bytes=%d active_cols=%d index=%d actual=%x expected=%x\n",
                       (int)stride_bytes, (int)active_cols, (int)i, actual,
                       expected);
              ++total_mismatches;
            }
          }
        }
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_active_cols_multirow_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_active_cols_multirow_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && zero_mismatches == 0 &&
       sentinel_mismatches == 0 && launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_active_cols_multirow_ssavc4 status=%s total_mismatches=%d zero_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)zero_mismatches,
         (int)sentinel_mismatches, (int)launch_failures,
         (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, full_dev);
  vc4Free(program, sparse_dev);
}
