#include "rpi.h"
#include "kernel_launch.h"

#define INPUT_ROWS 8u
#define ROWS 4u
#define COLS 8u
#define MAX_PITCH_WORDS 19u
#define INPUT_WORDS (((INPUT_ROWS - 1u) * MAX_PITCH_WORDS) + COLS)
#define VDR_OUTPUT_WORDS (((COLS - 1u) * MAX_PITCH_WORDS) + ROWS)
#define VDW_OUTPUT_WORDS (((ROWS - 1u) * MAX_PITCH_WORDS) + COLS)
#define GUARD_WORDS 16u
#define SENTINEL 0x8accee31u

static uint32_t input_values[INPUT_WORDS + GUARD_WORDS];
static uint32_t vdr_output[VDR_OUTPUT_WORDS + GUARD_WORDS];
static uint32_t vdw_output[VDW_OUTPUT_WORDS + GUARD_WORDS];
static const uint32_t active_rows_cases[] = {0u, 1u, 4u, 5u};
static const uint32_t active_cols_cases[] = {0u, 3u, 8u, 12u};
static const uint32_t pitch_stride_bytes_cases[] = {64u, 76u};

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x76000000u | (row << 8) | col;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t vdr_output_dev = 0;
  vc4_deviceptr_t vdw_output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vertical_rect_roundtrip_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &vdr_output_dev, sizeof(vdr_output)) < 0 ||
      vc4Malloc(program, &vdw_output_dev, sizeof(vdw_output)) < 0)
    panic("dynamic_vertical_rect_roundtrip_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t p = 0; p < sizeof(pitch_stride_bytes_cases) /
                               sizeof(pitch_stride_bytes_cases[0]); ++p) {
    uint32_t pitch_stride_bytes = pitch_stride_bytes_cases[p];
    uint32_t pitch_stride_words = pitch_stride_bytes / 4u;
    for (uint32_t r = 0;
         r < sizeof(active_rows_cases) / sizeof(active_rows_cases[0]); ++r) {
      uint32_t active_rows = active_rows_cases[r];
      uint32_t expected_rows = active_rows < ROWS ? active_rows : ROWS;
      for (uint32_t c = 0;
           c < sizeof(active_cols_cases) / sizeof(active_cols_cases[0]); ++c) {
        uint32_t active_cols = active_cols_cases[c];
        uint32_t expected_cols = active_cols < COLS ? active_cols : COLS;
        for (uint32_t i = 0; i < INPUT_WORDS + GUARD_WORDS; ++i)
          input_values[i] = SENTINEL;
        for (uint32_t i = 0; i < VDR_OUTPUT_WORDS + GUARD_WORDS; ++i)
          vdr_output[i] = SENTINEL;
        for (uint32_t i = 0; i < VDW_OUTPUT_WORDS + GUARD_WORDS; ++i)
          vdw_output[i] = SENTINEL;
        for (uint32_t row = 0; row < INPUT_ROWS; ++row)
          for (uint32_t col = 0; col < COLS; ++col)
            input_values[row * pitch_stride_words + col] = pattern(row, col);

        if (vc4MemcpyHtoD(program, input_dev, input_values,
                          sizeof(input_values)) < 0 ||
            vc4MemcpyHtoD(program, vdr_output_dev, vdr_output,
                          sizeof(vdr_output)) < 0 ||
            vc4MemcpyHtoD(program, vdw_output_dev, vdw_output,
                          sizeof(vdw_output)) < 0 ||
            dynamic_vertical_rect_roundtrip_ssavc4_launch(
                program, grid, block, input_dev, vdr_output_dev,
                vdw_output_dev, active_rows, active_cols,
                pitch_stride_bytes) < 0 ||
            vc4MemcpyDtoH(program, vdr_output, vdr_output_dev,
                          sizeof(vdr_output)) < 0 ||
            vc4MemcpyDtoH(program, vdw_output, vdw_output_dev,
                          sizeof(vdw_output)) < 0) {
          printk("ERROR: dynamic_vertical_rect_roundtrip launch/copy failed pitch_stride_bytes=%d active_rows=%d active_cols=%d\n",
                 (int)pitch_stride_bytes, (int)active_rows,
                 (int)active_cols);
          ++launch_failures;
        }

        for (uint32_t i = 0; i < VDR_OUTPUT_WORDS + GUARD_WORDS; ++i) {
          uint32_t expected_vdr = SENTINEL;
          for (uint32_t row = 0; row < expected_cols; ++row) {
            uint32_t row_base = row * pitch_stride_words;
            if (i >= row_base && i < row_base + expected_rows) {
              uint32_t col = i - row_base;
              expected_vdr = pattern(col, row);
              break;
            }
          }
          uint32_t actual_vdr = vdr_output[i];
          if (expected_vdr != SENTINEL)
            checksum += actual_vdr;
          if (actual_vdr != expected_vdr) {
            if (expected_vdr == SENTINEL) {
              if (sentinel_mismatches < 8)
                printk("ERROR: dynamic_vertical_rect_roundtrip vdr_sentinel pitch_stride_bytes=%d active_rows=%d active_cols=%d index=%d actual=%x\n",
                       (int)pitch_stride_bytes, (int)active_rows,
                       (int)active_cols, (int)i, actual_vdr);
              ++sentinel_mismatches;
            } else {
              if (total_mismatches < 8)
                printk("ERROR: dynamic_vertical_rect_roundtrip vdr_value pitch_stride_bytes=%d active_rows=%d active_cols=%d index=%d actual=%x expected=%x\n",
                       (int)pitch_stride_bytes, (int)active_rows,
                       (int)active_cols, (int)i, actual_vdr, expected_vdr);
              ++total_mismatches;
            }
          }
        }
        for (uint32_t i = 0; i < VDW_OUTPUT_WORDS + GUARD_WORDS; ++i) {
          uint32_t expected_vdw = SENTINEL;
          for (uint32_t row = 0; row < expected_rows; ++row) {
            uint32_t row_base = row * pitch_stride_words;
            if (i >= row_base && i < row_base + expected_cols) {
              uint32_t col = i - row_base;
              expected_vdw = pattern(col, row);
              break;
            }
          }
          uint32_t actual_vdw = vdw_output[i];
          if (expected_vdw != SENTINEL)
            checksum += actual_vdw;
          if (actual_vdw != expected_vdw) {
            if (expected_vdw == SENTINEL) {
              if (sentinel_mismatches < 8)
                printk("ERROR: dynamic_vertical_rect_roundtrip vdw_sentinel pitch_stride_bytes=%d active_rows=%d active_cols=%d index=%d actual=%x\n",
                       (int)pitch_stride_bytes, (int)active_rows,
                       (int)active_cols, (int)i, actual_vdw);
              ++sentinel_mismatches;
            } else {
              if (total_mismatches < 8)
                printk("ERROR: dynamic_vertical_rect_roundtrip vdw_value pitch_stride_bytes=%d active_rows=%d active_cols=%d index=%d actual=%x expected=%x\n",
                       (int)pitch_stride_bytes, (int)active_rows,
                       (int)active_cols, (int)i, actual_vdw, expected_vdw);
              ++total_mismatches;
            }
          }
        }
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_vertical_rect_roundtrip_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vertical_rect_roundtrip_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vertical_rect_roundtrip_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, vdr_output_dev);
  vc4Free(program, vdw_output_dev);
}
