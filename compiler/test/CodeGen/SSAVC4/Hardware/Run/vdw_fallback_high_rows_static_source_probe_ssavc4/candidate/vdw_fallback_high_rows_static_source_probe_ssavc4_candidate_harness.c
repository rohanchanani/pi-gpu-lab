#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 16u
#define STRIDE_BYTES 17000u
#define GUARD_BYTES 128u
#define OUTPUT_BYTES ((ROWS - 1u) * STRIDE_BYTES + 4u + GUARD_BYTES)
#define SENTINEL_BYTE 0x5au

static uint8_t output_bytes[OUTPUT_BYTES];

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x7e000000u | (row << 8) | col;
}

static uint8_t pattern_byte(uint32_t row, uint32_t col, uint32_t byte_index) {
  uint32_t value = pattern(row, col);
  return (uint8_t)(value >> (byte_index * 8u));
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, sizeof(output_bytes) + 4096u) < 0 ||
      !program)
    panic("vdw_fallback_high_rows_static_source_probe_ssavc4 program create failed");
  if (vc4Malloc(program, &output_dev, sizeof(output_bytes)) < 0)
    panic("vdw_fallback_high_rows_static_source_probe_ssavc4 allocation failed");

  for (uint32_t i = 0; i < sizeof(output_bytes); ++i)
    output_bytes[i] = SENTINEL_BYTE;

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  if (vc4MemcpyHtoD(program, output_dev, output_bytes,
                    sizeof(output_bytes)) < 0 ||
      vdw_fallback_high_rows_static_source_probe_ssavc4_launch(
          program, grid, block, output_dev) < 0 ||
      vc4MemcpyDtoH(program, output_bytes, output_dev,
                    sizeof(output_bytes)) < 0) {
    printk("ERROR: vdw_fallback_high_rows_static_source_probe launch/copy failed\n");
    ++launch_failures;
  }

  for (uint32_t i = 0; i < sizeof(output_bytes); ++i) {
    uint8_t expected = SENTINEL_BYTE;
    for (uint32_t row = 0; row < ROWS; ++row) {
      uint32_t row_base = row * STRIDE_BYTES;
      if (i >= row_base && i < row_base + 4u) {
        expected = pattern_byte(row, 0u, i - row_base);
        break;
      }
    }
    uint8_t actual = output_bytes[i];
    if (expected != SENTINEL_BYTE)
      checksum += actual;
    if (actual != expected) {
      if (expected == SENTINEL_BYTE) {
        if (sentinel_mismatches < 8)
          printk("ERROR: vdw_fallback_high_rows_static_source_probe sentinel byte=%d actual=%x expected=%x\n",
                 (int)i, actual, SENTINEL_BYTE);
        ++sentinel_mismatches;
      } else {
        if (total_mismatches < 8) {
          uint32_t row = i / STRIDE_BYTES;
          uint32_t byte_offset = i - row * STRIDE_BYTES;
          printk("ERROR: vdw_fallback_high_rows_static_source_probe row=%d byte_offset=%d byte_index=%d actual=%x expected=%x\n",
                 (int)row, (int)byte_offset, (int)i, actual, expected);
        }
        ++total_mismatches;
      }
    }
  }

  uint32_t runtime_launches =
      vdw_fallback_high_rows_static_source_probe_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      vdw_fallback_high_rows_static_source_probe_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_fallback_high_rows_static_source_probe_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, output_dev);
}
