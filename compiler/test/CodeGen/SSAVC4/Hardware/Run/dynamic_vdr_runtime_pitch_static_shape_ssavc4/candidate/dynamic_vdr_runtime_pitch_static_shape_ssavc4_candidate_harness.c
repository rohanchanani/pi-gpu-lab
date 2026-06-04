#include "rpi.h"
#include "kernel_launch.h"

#define MAX_ROWS 16u
#define MAX_COLS 16u
#define MAX_PITCH_WORDS 65u
#define IN_WORDS ((MAX_ROWS - 1u) * MAX_PITCH_WORDS + MAX_COLS)
#define OUT_WORDS (MAX_ROWS * MAX_COLS)
#define GUARD_WORDS 16u
#define SENTINEL 0x8accee31u

struct shape_case {
  const char *name;
  uint32_t rows;
  uint32_t cols;
  uint32_t tag;
};

static const struct shape_case shapes[] = {
    {"16x16", 16u, 16u, 0x10u},
    {"8x16", 8u, 16u, 0x20u},
    {"16x8", 16u, 8u, 0x30u},
    {"7x5", 7u, 5u, 0x40u},
};
static const uint32_t pitch_bytes_cases[] = {64u, 76u, 124u, 260u};

static uint32_t input_16x16[IN_WORDS + GUARD_WORDS];
static uint32_t input_8x16[IN_WORDS + GUARD_WORDS];
static uint32_t input_16x8[IN_WORDS + GUARD_WORDS];
static uint32_t input_7x5[IN_WORDS + GUARD_WORDS];
static uint32_t output_16x16[OUT_WORDS + GUARD_WORDS];
static uint32_t output_8x16[OUT_WORDS + GUARD_WORDS];
static uint32_t output_16x8[OUT_WORDS + GUARD_WORDS];
static uint32_t output_7x5[OUT_WORDS + GUARD_WORDS];

static uint32_t pattern(uint32_t tag, uint32_t row, uint32_t col) {
  return 0x72000000u | (tag << 16) | (row << 8) | col;
}

static void fill_input(uint32_t *input, const struct shape_case *shape,
                       uint32_t pitch_words) {
  for (uint32_t i = 0; i < IN_WORDS + GUARD_WORDS; ++i)
    input[i] = SENTINEL;
  for (uint32_t row = 0; row < shape->rows; ++row)
    for (uint32_t col = 0; col < shape->cols; ++col)
      input[row * pitch_words + col] = pattern(shape->tag, row, col);
}

static void fill_output(uint32_t *output) {
  for (uint32_t i = 0; i < OUT_WORDS + GUARD_WORDS; ++i)
    output[i] = SENTINEL;
}

static void verify_output(const uint32_t *out_values,
                          const struct shape_case *shape,
                          uint32_t pitch_bytes, uint32_t *total_mismatches,
                          uint32_t *zero_mismatches,
                          uint32_t *sentinel_mismatches,
                          uint32_t *checksum) {
  for (uint32_t row = 0; row < MAX_ROWS; ++row) {
    for (uint32_t col = 0; col < MAX_COLS; ++col) {
      uint32_t actual = out_values[row * MAX_COLS + col];
      uint32_t expected =
          row < shape->rows && col < shape->cols
              ? pattern(shape->tag, row, col)
              : 0u;
      if (row < shape->rows && col < shape->cols)
        *checksum += actual;
      if (actual != expected) {
        if (expected != 0u) {
          if (*total_mismatches < 8)
            printk("ERROR: dynamic_vdr_runtime_pitch_static_shape shape=%s pitch_bytes=%d row=%d col=%d actual=%x expected=%x\n",
                   shape->name, (int)pitch_bytes, (int)row, (int)col, actual,
                   expected);
          ++*total_mismatches;
        } else {
          if (*zero_mismatches < 8)
            printk("ERROR: dynamic_vdr_runtime_pitch_static_shape shape=%s pitch_bytes=%d zero_row=%d zero_col=%d actual=%x expected=0\n",
                   shape->name, (int)pitch_bytes, (int)row, (int)col, actual);
          ++*zero_mismatches;
        }
      }
    }
  }
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    uint32_t actual = out_values[OUT_WORDS + i];
    if (actual != SENTINEL) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: dynamic_vdr_runtime_pitch_static_shape shape=%s pitch_bytes=%d guard=%d actual=%x expected=%x\n",
               shape->name, (int)pitch_bytes, (int)i, actual, SENTINEL);
      ++*sentinel_mismatches;
    }
  }
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_16x16_dev = 0;
  vc4_deviceptr_t in_8x16_dev = 0;
  vc4_deviceptr_t in_16x8_dev = 0;
  vc4_deviceptr_t in_7x5_dev = 0;
  vc4_deviceptr_t out_16x16_dev = 0;
  vc4_deviceptr_t out_8x16_dev = 0;
  vc4_deviceptr_t out_16x8_dev = 0;
  vc4_deviceptr_t out_7x5_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t zero_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdr_runtime_pitch_static_shape_ssavc4 program create failed");
  if (vc4Malloc(program, &in_16x16_dev, sizeof(input_16x16)) < 0 ||
      vc4Malloc(program, &in_8x16_dev, sizeof(input_8x16)) < 0 ||
      vc4Malloc(program, &in_16x8_dev, sizeof(input_16x8)) < 0 ||
      vc4Malloc(program, &in_7x5_dev, sizeof(input_7x5)) < 0 ||
      vc4Malloc(program, &out_16x16_dev, sizeof(output_16x16)) < 0 ||
      vc4Malloc(program, &out_8x16_dev, sizeof(output_8x16)) < 0 ||
      vc4Malloc(program, &out_16x8_dev, sizeof(output_16x8)) < 0 ||
      vc4Malloc(program, &out_7x5_dev, sizeof(output_7x5)) < 0)
    panic("dynamic_vdr_runtime_pitch_static_shape_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t p = 0;
       p < sizeof(pitch_bytes_cases) / sizeof(pitch_bytes_cases[0]); ++p) {
    uint32_t pitch_bytes = pitch_bytes_cases[p];
    uint32_t pitch_words = pitch_bytes / 4u;
    fill_input(input_16x16, &shapes[0], pitch_words);
    fill_input(input_8x16, &shapes[1], pitch_words);
    fill_input(input_16x8, &shapes[2], pitch_words);
    fill_input(input_7x5, &shapes[3], pitch_words);
    fill_output(output_16x16);
    fill_output(output_8x16);
    fill_output(output_16x8);
    fill_output(output_7x5);

    if (vc4MemcpyHtoD(program, in_16x16_dev, input_16x16,
                      sizeof(input_16x16)) < 0 ||
        vc4MemcpyHtoD(program, in_8x16_dev, input_8x16,
                      sizeof(input_8x16)) < 0 ||
        vc4MemcpyHtoD(program, in_16x8_dev, input_16x8,
                      sizeof(input_16x8)) < 0 ||
        vc4MemcpyHtoD(program, in_7x5_dev, input_7x5,
                      sizeof(input_7x5)) < 0 ||
        vc4MemcpyHtoD(program, out_16x16_dev, output_16x16,
                      sizeof(output_16x16)) < 0 ||
        vc4MemcpyHtoD(program, out_8x16_dev, output_8x16,
                      sizeof(output_8x16)) < 0 ||
        vc4MemcpyHtoD(program, out_16x8_dev, output_16x8,
                      sizeof(output_16x8)) < 0 ||
        vc4MemcpyHtoD(program, out_7x5_dev, output_7x5,
                      sizeof(output_7x5)) < 0 ||
        dynamic_vdr_runtime_pitch_static_shape_ssavc4_launch(
            program, grid, block, in_16x16_dev, in_8x16_dev, in_16x8_dev,
            in_7x5_dev, out_16x16_dev, out_8x16_dev, out_16x8_dev,
            out_7x5_dev, pitch_bytes) < 0 ||
        vc4MemcpyDtoH(program, output_16x16, out_16x16_dev,
                      sizeof(output_16x16)) < 0 ||
        vc4MemcpyDtoH(program, output_8x16, out_8x16_dev,
                      sizeof(output_8x16)) < 0 ||
        vc4MemcpyDtoH(program, output_16x8, out_16x8_dev,
                      sizeof(output_16x8)) < 0 ||
        vc4MemcpyDtoH(program, output_7x5, out_7x5_dev,
                      sizeof(output_7x5)) < 0) {
      printk("ERROR: dynamic_vdr_runtime_pitch_static_shape launch/copy failed pitch_bytes=%d\n",
             (int)pitch_bytes);
      ++launch_failures;
    }

    verify_output(output_16x16, &shapes[0], pitch_bytes, &total_mismatches,
                  &zero_mismatches, &sentinel_mismatches, &checksum);
    verify_output(output_8x16, &shapes[1], pitch_bytes, &total_mismatches,
                  &zero_mismatches, &sentinel_mismatches, &checksum);
    verify_output(output_16x8, &shapes[2], pitch_bytes, &total_mismatches,
                  &zero_mismatches, &sentinel_mismatches, &checksum);
    verify_output(output_7x5, &shapes[3], pitch_bytes, &total_mismatches,
                  &zero_mismatches, &sentinel_mismatches, &checksum);
  }

  uint32_t runtime_launches =
      dynamic_vdr_runtime_pitch_static_shape_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdr_runtime_pitch_static_shape_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && zero_mismatches == 0 &&
       sentinel_mismatches == 0 && launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdr_runtime_pitch_static_shape_ssavc4 status=%s total_mismatches=%d zero_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)zero_mismatches,
         (int)sentinel_mismatches, (int)launch_failures,
         (int)runtime_launches, checksum);

  vc4Free(program, in_16x16_dev);
  vc4Free(program, in_8x16_dev);
  vc4Free(program, in_16x8_dev);
  vc4Free(program, in_7x5_dev);
  vc4Free(program, out_16x16_dev);
  vc4Free(program, out_8x16_dev);
  vc4Free(program, out_16x8_dev);
  vc4Free(program, out_7x5_dev);
}
