#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 16u
#define COLS 16u
#define MAX_OUTPUT_BYTES ((ROWS - 1u) * 17000u + COLS * 4u + GUARD_BYTES)
#define GUARD_BYTES 128u
#define SENTINEL_BYTE 0x5au

static uint32_t input_values[ROWS * COLS];
static uint8_t output_bytes[MAX_OUTPUT_BYTES];
static const uint32_t active_rows_cases[] = {0u, 1u, 7u, 16u};
static const uint32_t active_cols_cases[] = {0u, 1u, 7u, 16u};
static const uint32_t stride_bytes_cases[] = {64u, 68u, 8191u, 8192u, 17000u};

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0xc3000000u | (row << 8) | col;
}

static uint8_t pattern_byte(uint32_t row, uint32_t col, uint32_t byte_index) {
  uint32_t value = pattern(row, col);
  return (uint8_t)(value >> (byte_index * 8u));
}

static uint32_t required_output_bytes(uint32_t rows, uint32_t stride_bytes) {
  if (rows == 0)
    return COLS * sizeof(uint32_t) + GUARD_BYTES;
  return (rows - 1u) * stride_bytes + COLS * sizeof(uint32_t) + GUARD_BYTES;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;
  const uint32_t output_byte_count =
      (uint32_t)(sizeof(output_bytes) / sizeof(output_bytes[0]));

  const uint32_t program_heap_bytes = sizeof(input_values) + sizeof(output_bytes) + 4096u;
  if (vc4_program_create(&program, program_heap_bytes) < 0 || !program)
    panic("dynamic_vdw_stride_13bit_boundary_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &output_dev, sizeof(output_bytes)) < 0)
    panic("dynamic_vdw_stride_13bit_boundary_ssavc4 allocation failed");

  for (uint32_t row = 0; row < ROWS; ++row)
    for (uint32_t col = 0; col < COLS; ++col)
      input_values[row * COLS + col] = pattern(row, col);

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t s = 0;
       s < sizeof(stride_bytes_cases) / sizeof(stride_bytes_cases[0]); ++s) {
    uint32_t stride_bytes = stride_bytes_cases[s];
    for (uint32_t r = 0;
         r < sizeof(active_rows_cases) / sizeof(active_rows_cases[0]); ++r) {
      uint32_t active_rows = active_rows_cases[r];
      uint32_t expected_rows = active_rows < ROWS ? active_rows : ROWS;
      if (required_output_bytes(expected_rows, stride_bytes) >
          output_byte_count)
        continue;
      for (uint32_t c = 0;
           c < sizeof(active_cols_cases) / sizeof(active_cols_cases[0]); ++c) {
        uint32_t active_cols = active_cols_cases[c];
        uint32_t expected_cols = active_cols < COLS ? active_cols : COLS;
        for (uint32_t i = 0; i < output_byte_count; ++i)
          output_bytes[i] = SENTINEL_BYTE;

        if (vc4MemcpyHtoD(program, input_dev, input_values,
                          sizeof(input_values)) < 0 ||
            vc4MemcpyHtoD(program, output_dev, output_bytes,
                          sizeof(output_bytes)) < 0 ||
            dynamic_vdw_stride_13bit_boundary_ssavc4_launch(
                program, grid, block, input_dev, output_dev, active_rows,
                active_cols, stride_bytes) < 0 ||
            vc4MemcpyDtoH(program, output_bytes, output_dev,
                          sizeof(output_bytes)) < 0) {
          printk("ERROR: dynamic_vdw_stride_13bit_boundary launch/copy failed stride_bytes=%d active_rows=%d active_cols=%d\n",
                 (int)stride_bytes, (int)active_rows, (int)active_cols);
          ++launch_failures;
        }

        for (uint32_t i = 0; i < output_byte_count; ++i) {
          uint8_t expected = SENTINEL_BYTE;
          uint32_t expected_row = 0xffffffffu;
          uint32_t expected_col = 0xffffffffu;
          uint32_t expected_byte_offset = 0xffffffffu;
          for (uint32_t row = 0; row < expected_rows; ++row) {
            uint32_t row_base = row * stride_bytes;
            uint32_t row_bytes = expected_cols * sizeof(uint32_t);
            if (i >= row_base && i < row_base + row_bytes) {
              uint32_t offset = i - row_base;
              expected_row = row;
              expected_col = offset / sizeof(uint32_t);
              expected_byte_offset = offset % sizeof(uint32_t);
              expected = pattern_byte(row, expected_col, expected_byte_offset);
              break;
            }
          }
          uint8_t actual = output_bytes[i];
          if (expected != SENTINEL_BYTE)
            checksum += actual;
          if (actual != expected) {
            if (expected != SENTINEL_BYTE) {
              if (total_mismatches < 8)
                printk("ERROR: dynamic_vdw_stride_13bit_boundary stride_bytes=%d active_rows=%d active_cols=%d row=%d col=%d byte_offset=%d byte_index=%d actual=%x expected=%x\n",
                       (int)stride_bytes, (int)active_rows,
                       (int)active_cols, (int)expected_row,
                       (int)expected_col, (int)expected_byte_offset, (int)i,
                       actual, expected);
              ++total_mismatches;
            } else {
              uint32_t observed_row =
                  stride_bytes == 0 ? 0xffffffffu : i / stride_bytes;
              uint32_t observed_row_offset =
                  stride_bytes == 0 ? i : i - observed_row * stride_bytes;
              uint32_t observed_col = observed_row_offset / sizeof(uint32_t);
              uint32_t observed_byte_offset =
                  observed_row_offset % sizeof(uint32_t);
              if (sentinel_mismatches < 8)
                printk("ERROR: dynamic_vdw_stride_13bit_boundary stride_bytes=%d active_rows=%d active_cols=%d row=%d col=%d byte_offset=%d sentinel_byte=%d actual=%x expected=%x\n",
                       (int)stride_bytes, (int)active_rows,
                       (int)active_cols, (int)observed_row,
                       (int)observed_col, (int)observed_byte_offset, (int)i,
                       actual, SENTINEL_BYTE);
              ++sentinel_mismatches;
            }
          }
        }
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_vdw_stride_13bit_boundary_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdw_stride_13bit_boundary_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdw_stride_13bit_boundary_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
