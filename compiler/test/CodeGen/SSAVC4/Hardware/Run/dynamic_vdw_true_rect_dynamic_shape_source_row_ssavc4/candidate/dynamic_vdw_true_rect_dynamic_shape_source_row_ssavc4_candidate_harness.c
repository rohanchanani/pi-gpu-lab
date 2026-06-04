#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 64u
#define BAND_ROWS 16u
#define COLS 16u
#define GUARD_BYTES 128u
#define MAX_STRIDE_BYTES 17000u
#define MAX_OUTPUT_BYTES ((16u - 1u) * MAX_STRIDE_BYTES + COLS * 4u + GUARD_BYTES)
#define SENTINEL_BYTE 0x5au

static uint32_t input0[BAND_ROWS * COLS];
static uint32_t input16[BAND_ROWS * COLS];
static uint32_t input32[BAND_ROWS * COLS];
static uint32_t input48[BAND_ROWS * COLS];
static uint8_t output_bytes[MAX_OUTPUT_BYTES];
static uint8_t preserve_bytes[MAX_OUTPUT_BYTES];

struct dynamic_case {
  uint32_t source_row;
  uint32_t active_rows;
  uint32_t active_cols;
  uint32_t stride_bytes;
};

static const struct dynamic_case cases[] = {
    {0u, 0u, 0u, 64u},       {0u, 0u, 16u, 8192u},
    {1u, 7u, 0u, 68u},       {15u, 16u, 0u, 17000u},
    {0u, 1u, 1u, 64u},       {0u, 7u, 1u, 64u},
    {0u, 16u, 16u, 64u},     {1u, 7u, 5u, 68u},
    {15u, 16u, 5u, 68u},     {16u, 1u, 16u, 8192u},
    {16u, 7u, 1u, 8192u},    {16u, 16u, 16u, 68u},
    {31u, 1u, 5u, 64u},      {31u, 16u, 16u, 8192u},
    {32u, 7u, 5u, 17000u},   {32u, 16u, 1u, 17000u},
    {48u, 1u, 16u, 17000u},  {48u, 7u, 5u, 68u},
    {48u, 16u, 1u, 64u},     {63u, 1u, 1u, 64u},
    {63u, 1u, 16u, 68u},     {15u, 1u, 5u, 8192u},
    {32u, 1u, 0u, 8192u},    {48u, 0u, 5u, 17000u},
    {1u, 16u, 1u, 68u},      {15u, 7u, 16u, 64u},
    {31u, 7u, 5u, 17000u},   {32u, 16u, 16u, 8192u},
};

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0xd4000000u | (row << 8) | col;
}

static uint8_t pattern_byte(uint32_t row, uint32_t col, uint32_t byte_index) {
  uint32_t value = pattern(row, col);
  return (uint8_t)(value >> (byte_index * 8u));
}

static void fill_band(uint32_t *values, uint32_t base_row) {
  for (uint32_t row = 0; row < BAND_ROWS; ++row)
    for (uint32_t col = 0; col < COLS; ++col)
      values[row * COLS + col] = pattern(base_row + row, col);
}

static void fill_bytes(uint8_t *values) {
  for (uint32_t i = 0; i < MAX_OUTPUT_BYTES; ++i)
    values[i] = SENTINEL_BYTE;
}

static uint32_t required_output_bytes(uint32_t rows, uint32_t stride_bytes) {
  if (rows == 0)
    return COLS * sizeof(uint32_t) + GUARD_BYTES;
  return (rows - 1u) * stride_bytes + COLS * sizeof(uint32_t) + GUARD_BYTES;
}

static uint8_t expected_main_byte(uint32_t byte_index, uint32_t source_row,
                                  uint32_t active_rows,
                                  uint32_t active_cols,
                                  uint32_t stride_bytes) {
  for (uint32_t row = 0; row < active_rows; ++row) {
    uint32_t row_base = row * stride_bytes;
    uint32_t row_bytes = active_cols * sizeof(uint32_t);
    if (byte_index >= row_base && byte_index < row_base + row_bytes) {
      uint32_t offset = byte_index - row_base;
      return pattern_byte(source_row + row, offset / sizeof(uint32_t),
                          offset % sizeof(uint32_t));
    }
  }
  return SENTINEL_BYTE;
}

static uint8_t expected_preserve_byte(uint32_t byte_index, uint32_t source_row,
                                      uint32_t active_rows,
                                      uint32_t active_cols,
                                      uint32_t stride_bytes) {
  uint8_t expected =
      byte_index < COLS * sizeof(uint32_t)
          ? pattern_byte(0u, byte_index / sizeof(uint32_t),
                         byte_index % sizeof(uint32_t))
          : SENTINEL_BYTE;
  for (uint32_t row = 0; row < active_rows; ++row) {
    uint32_t row_base = row * stride_bytes;
    uint32_t row_bytes = active_cols * sizeof(uint32_t);
    if (byte_index >= row_base && byte_index < row_base + row_bytes) {
      uint32_t offset = byte_index - row_base;
      expected = pattern_byte(source_row + row, offset / sizeof(uint32_t),
                              offset % sizeof(uint32_t));
      break;
    }
  }
  return expected;
}

static void check_buffer(const char *label, const uint8_t *actual_bytes,
                         uint32_t source_row, uint32_t active_rows,
                         uint32_t active_cols, uint32_t stride_bytes,
                         uint32_t check_bytes, int preserve_mode,
                         uint32_t *total_mismatches,
                         uint32_t *sentinel_mismatches, uint32_t *checksum) {
  for (uint32_t i = 0; i < check_bytes; ++i) {
    uint8_t expected =
        preserve_mode
            ? expected_preserve_byte(i, source_row, active_rows, active_cols,
                                     stride_bytes)
            : expected_main_byte(i, source_row, active_rows, active_cols,
                                 stride_bytes);
    uint8_t actual = actual_bytes[i];
    if (expected != SENTINEL_BYTE)
      *checksum += actual;
    if (actual != expected) {
      if (expected == SENTINEL_BYTE) {
        if (*sentinel_mismatches < 8)
          printk("ERROR: dynamic_vdw_true_rect %s source_row=%d active_rows=%d active_cols=%d stride=%d sentinel_byte=%d actual=%x expected=%x\n",
                 label, (int)source_row, (int)active_rows, (int)active_cols,
                 (int)stride_bytes, (int)i, actual, SENTINEL_BYTE);
        ++*sentinel_mismatches;
      } else {
        if (*total_mismatches < 8)
          printk("ERROR: dynamic_vdw_true_rect %s source_row=%d active_rows=%d active_cols=%d stride=%d byte=%d actual=%x expected=%x\n",
                 label, (int)source_row, (int)active_rows, (int)active_cols,
                 (int)stride_bytes, (int)i, actual, expected);
        ++*total_mismatches;
      }
    }
  }
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input0_dev = 0;
  vc4_deviceptr_t input16_dev = 0;
  vc4_deviceptr_t input32_dev = 0;
  vc4_deviceptr_t input48_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  vc4_deviceptr_t preserve_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;
  uint32_t launched_cases = 0;

  uint32_t heap_bytes = sizeof(input0) + sizeof(input16) + sizeof(input32) +
                        sizeof(input48) + sizeof(output_bytes) +
                        sizeof(preserve_bytes) + 4096u;
  if (vc4_program_create(&program, heap_bytes) < 0 || !program)
    panic("dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4 program create failed");
  if (vc4Malloc(program, &input0_dev, sizeof(input0)) < 0 ||
      vc4Malloc(program, &input16_dev, sizeof(input16)) < 0 ||
      vc4Malloc(program, &input32_dev, sizeof(input32)) < 0 ||
      vc4Malloc(program, &input48_dev, sizeof(input48)) < 0 ||
      vc4Malloc(program, &output_dev, sizeof(output_bytes)) < 0 ||
      vc4Malloc(program, &preserve_dev, sizeof(preserve_bytes)) < 0)
    panic("dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4 allocation failed");

  fill_band(input0, 0u);
  fill_band(input16, 16u);
  fill_band(input32, 32u);
  fill_band(input48, 48u);

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
    uint32_t source_row = cases[ci].source_row;
    uint32_t active_rows = cases[ci].active_rows;
    uint32_t active_cols = cases[ci].active_cols;
    uint32_t stride_bytes = cases[ci].stride_bytes;
    uint32_t expected_rows = active_rows < 16u ? active_rows : 16u;
    uint32_t expected_cols = active_cols < COLS ? active_cols : COLS;
    if (source_row + expected_rows > ROWS)
      panic("dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4 illegal case");
    uint32_t check_bytes = required_output_bytes(expected_rows, stride_bytes);
    fill_bytes(output_bytes);
    fill_bytes(preserve_bytes);

    if (vc4MemcpyHtoD(program, input0_dev, input0, sizeof(input0)) < 0 ||
        vc4MemcpyHtoD(program, input16_dev, input16, sizeof(input16)) < 0 ||
        vc4MemcpyHtoD(program, input32_dev, input32, sizeof(input32)) < 0 ||
        vc4MemcpyHtoD(program, input48_dev, input48, sizeof(input48)) < 0 ||
        vc4MemcpyHtoD(program, output_dev, output_bytes,
                      sizeof(output_bytes)) < 0 ||
        vc4MemcpyHtoD(program, preserve_dev, preserve_bytes,
                      sizeof(preserve_bytes)) < 0 ||
        dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4_launch(
            program, grid, block, input0_dev, input16_dev, input32_dev,
            input48_dev, output_dev, preserve_dev, source_row, active_rows,
            active_cols, stride_bytes) < 0 ||
        vc4MemcpyDtoH(program, output_bytes, output_dev,
                      sizeof(output_bytes)) < 0 ||
        vc4MemcpyDtoH(program, preserve_bytes, preserve_dev,
                      sizeof(preserve_bytes)) < 0) {
      printk("ERROR: dynamic_vdw_true_rect launch/copy failed source_row=%d active_rows=%d active_cols=%d stride=%d\n",
             (int)source_row, (int)active_rows, (int)active_cols,
             (int)stride_bytes);
      ++launch_failures;
    } else {
      check_buffer("main", output_bytes, source_row, expected_rows,
                   expected_cols, stride_bytes, check_bytes,
                   /*preserve_mode=*/0, &total_mismatches,
                   &sentinel_mismatches, &checksum);
      check_buffer("preserve", preserve_bytes, source_row, expected_rows,
                   expected_cols, stride_bytes, check_bytes,
                   /*preserve_mode=*/1, &total_mismatches,
                   &sentinel_mismatches, &checksum);
    }
    ++launched_cases;
  }

  uint32_t runtime_launches =
      dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0 && runtime_launches == launched_cases)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)launched_cases, (int)total_mismatches,
         (int)sentinel_mismatches, (int)launch_failures,
         (int)runtime_launches, checksum);

  vc4Free(program, input0_dev);
  vc4Free(program, input16_dev);
  vc4Free(program, input32_dev);
  vc4Free(program, input48_dev);
  vc4Free(program, output_dev);
  vc4Free(program, preserve_dev);
}
