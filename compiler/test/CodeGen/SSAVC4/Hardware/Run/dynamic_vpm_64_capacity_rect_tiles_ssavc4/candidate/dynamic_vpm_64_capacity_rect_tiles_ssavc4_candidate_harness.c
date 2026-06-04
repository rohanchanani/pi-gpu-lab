#include "rpi.h"
#include "kernel_launch.h"

#define TILE_COUNT 4u
#define TILE_ROWS 16u
#define TILE_COLS 16u
#define TILE_WORDS (TILE_ROWS * TILE_COLS)
#define GUARD_WORDS 32u
#define BUFFER_WORDS (GUARD_WORDS + TILE_WORDS + GUARD_WORDS)
#define SENTINEL 0x9e5a77edu

static uint32_t input0[BUFFER_WORDS];
static uint32_t input16[BUFFER_WORDS];
static uint32_t input32[BUFFER_WORDS];
static uint32_t input48[BUFFER_WORDS];
static uint32_t output0[BUFFER_WORDS];
static uint32_t output16[BUFFER_WORDS];
static uint32_t output32[BUFFER_WORDS];
static uint32_t output48[BUFFER_WORDS];

static const uint32_t row_bases[TILE_COUNT] = {0u, 16u, 32u, 48u};

static uint32_t pattern(uint32_t tile_id, uint32_t vpm_row, uint32_t col) {
  return 0xb5000000u | ((tile_id & 0xfu) << 20) |
         ((vpm_row & 0xffu) << 8) | (col & 0xffu);
}

static void fill_source(uint32_t *buffer, uint32_t tile_id,
                        uint32_t base_row) {
  for (uint32_t i = 0; i < BUFFER_WORDS; ++i)
    buffer[i] = SENTINEL;
  for (uint32_t row = 0; row < TILE_ROWS; ++row)
    for (uint32_t col = 0; col < TILE_COLS; ++col)
      buffer[GUARD_WORDS + row * TILE_COLS + col] =
          pattern(tile_id, base_row + row, col);
}

static void fill_output(uint32_t *buffer) {
  for (uint32_t i = 0; i < BUFFER_WORDS; ++i)
    buffer[i] = SENTINEL;
}

static uint32_t check_source_guard(const char *label, const uint32_t *buffer) {
  uint32_t mismatches = 0;
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    if (buffer[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: dynamic_vpm_64_capacity %s source leading_guard=%d actual=%x expected=%x\n",
               label, (int)i, buffer[i], SENTINEL);
      ++mismatches;
    }
    if (buffer[GUARD_WORDS + TILE_WORDS + i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: dynamic_vpm_64_capacity %s source trailing_guard=%d actual=%x expected=%x\n",
               label, (int)i, buffer[GUARD_WORDS + TILE_WORDS + i], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

static void check_output_tile(const char *label, const uint32_t *buffer,
                              uint32_t tile_id, uint32_t base_row,
                              uint32_t *total_mismatches,
                              uint32_t *sentinel_mismatches,
                              uint32_t *checksum) {
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    if (buffer[i] != SENTINEL) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: dynamic_vpm_64_capacity %s leading_guard=%d actual=%x expected=%x\n",
               label, (int)i, buffer[i], SENTINEL);
      ++*sentinel_mismatches;
    }
    if (buffer[GUARD_WORDS + TILE_WORDS + i] != SENTINEL) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: dynamic_vpm_64_capacity %s trailing_guard=%d actual=%x expected=%x\n",
               label, (int)i, buffer[GUARD_WORDS + TILE_WORDS + i], SENTINEL);
      ++*sentinel_mismatches;
    }
  }

  for (uint32_t row = 0; row < TILE_ROWS; ++row) {
    for (uint32_t col = 0; col < TILE_COLS; ++col) {
      uint32_t actual = buffer[GUARD_WORDS + row * TILE_COLS + col];
      uint32_t expected = pattern(tile_id, base_row + row, col);
      *checksum += actual;
      if (actual != expected) {
        if (*total_mismatches < 16)
          printk("ERROR: dynamic_vpm_64_capacity %s row_base=%d row=%d col=%d actual=%x expected=%x\n",
                 label, (int)base_row, (int)row, (int)col, actual, expected);
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
  vc4_deviceptr_t output0_dev = 0;
  vc4_deviceptr_t output16_dev = 0;
  vc4_deviceptr_t output32_dev = 0;
  vc4_deviceptr_t output48_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t source_guard_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vpm_64_capacity_rect_tiles_ssavc4 program create failed");
  if (vc4Malloc(program, &input0_dev, sizeof(input0)) < 0 ||
      vc4Malloc(program, &input16_dev, sizeof(input16)) < 0 ||
      vc4Malloc(program, &input32_dev, sizeof(input32)) < 0 ||
      vc4Malloc(program, &input48_dev, sizeof(input48)) < 0 ||
      vc4Malloc(program, &output0_dev, sizeof(output0)) < 0 ||
      vc4Malloc(program, &output16_dev, sizeof(output16)) < 0 ||
      vc4Malloc(program, &output32_dev, sizeof(output32)) < 0 ||
      vc4Malloc(program, &output48_dev, sizeof(output48)) < 0)
    panic("dynamic_vpm_64_capacity_rect_tiles_ssavc4 allocation failed");

  fill_source(input0, 0u, row_bases[0]);
  fill_source(input16, 1u, row_bases[1]);
  fill_source(input32, 2u, row_bases[2]);
  fill_source(input48, 3u, row_bases[3]);
  fill_output(output0);
  fill_output(output16);
  fill_output(output32);
  fill_output(output48);

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  uint32_t active_rows = 16u;
  uint32_t active_cols = 16u;
  uint32_t pitch_bytes = TILE_COLS * sizeof(uint32_t);
  uint32_t stride_bytes = TILE_COLS * sizeof(uint32_t);

  if (vc4MemcpyHtoD(program, input0_dev, input0, sizeof(input0)) < 0 ||
      vc4MemcpyHtoD(program, input16_dev, input16, sizeof(input16)) < 0 ||
      vc4MemcpyHtoD(program, input32_dev, input32, sizeof(input32)) < 0 ||
      vc4MemcpyHtoD(program, input48_dev, input48, sizeof(input48)) < 0 ||
      vc4MemcpyHtoD(program, output0_dev, output0, sizeof(output0)) < 0 ||
      vc4MemcpyHtoD(program, output16_dev, output16, sizeof(output16)) < 0 ||
      vc4MemcpyHtoD(program, output32_dev, output32, sizeof(output32)) < 0 ||
      vc4MemcpyHtoD(program, output48_dev, output48, sizeof(output48)) < 0 ||
      dynamic_vpm_64_capacity_rect_tiles_ssavc4_launch(
          program, grid, block, input0_dev + GUARD_WORDS * sizeof(uint32_t),
          input16_dev + GUARD_WORDS * sizeof(uint32_t),
          input32_dev + GUARD_WORDS * sizeof(uint32_t),
          input48_dev + GUARD_WORDS * sizeof(uint32_t),
          output0_dev + GUARD_WORDS * sizeof(uint32_t),
          output16_dev + GUARD_WORDS * sizeof(uint32_t),
          output32_dev + GUARD_WORDS * sizeof(uint32_t),
          output48_dev + GUARD_WORDS * sizeof(uint32_t), active_rows,
          active_cols, pitch_bytes, stride_bytes) < 0 ||
      vc4MemcpyDtoH(program, input0, input0_dev, sizeof(input0)) < 0 ||
      vc4MemcpyDtoH(program, input16, input16_dev, sizeof(input16)) < 0 ||
      vc4MemcpyDtoH(program, input32, input32_dev, sizeof(input32)) < 0 ||
      vc4MemcpyDtoH(program, input48, input48_dev, sizeof(input48)) < 0 ||
      vc4MemcpyDtoH(program, output0, output0_dev, sizeof(output0)) < 0 ||
      vc4MemcpyDtoH(program, output16, output16_dev, sizeof(output16)) < 0 ||
      vc4MemcpyDtoH(program, output32, output32_dev, sizeof(output32)) < 0 ||
      vc4MemcpyDtoH(program, output48, output48_dev, sizeof(output48)) < 0) {
    printk("ERROR: dynamic_vpm_64_capacity launch/copy failed\n");
    ++launch_failures;
  } else {
    source_guard_mismatches += check_source_guard("tile0", input0);
    source_guard_mismatches += check_source_guard("tile16", input16);
    source_guard_mismatches += check_source_guard("tile32", input32);
    source_guard_mismatches += check_source_guard("tile48", input48);
    check_output_tile("tile0", output0, 0u, row_bases[0], &total_mismatches,
                      &sentinel_mismatches, &checksum);
    check_output_tile("tile16", output16, 1u, row_bases[1], &total_mismatches,
                      &sentinel_mismatches, &checksum);
    check_output_tile("tile32", output32, 2u, row_bases[2], &total_mismatches,
                      &sentinel_mismatches, &checksum);
    check_output_tile("tile48", output48, 3u, row_bases[3], &total_mismatches,
                      &sentinel_mismatches, &checksum);
  }

  uint32_t runtime_launches =
      dynamic_vpm_64_capacity_rect_tiles_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vpm_64_capacity_rect_tiles_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       source_guard_mismatches == 0 && launch_failures == 0 &&
       runtime_launches == 1)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vpm_64_capacity_rect_tiles_ssavc4 status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d source_guard_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u row_bases=0,16,32,48\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)source_guard_mismatches, (int)launch_failures,
         (int)runtime_launches, checksum);

  vc4Free(program, input0_dev);
  vc4Free(program, input16_dev);
  vc4Free(program, input32_dev);
  vc4Free(program, input48_dev);
  vc4Free(program, output0_dev);
  vc4Free(program, output16_dev);
  vc4Free(program, output32_dev);
  vc4Free(program, output48_dev);
}
