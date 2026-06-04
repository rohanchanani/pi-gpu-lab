#include "rpi.h"
#include "kernel_launch.h"

#define FIRST_WORDS 1u
#define SECOND_BYTES 1024u
#define GUARD_WORDS 16u
#define SENTINEL_WORD 0x5a5a5a5au
#define SENTINEL_BYTE 0x5au

static uint32_t first_words[FIRST_WORDS + GUARD_WORDS];
static uint8_t second_bytes[SECOND_BYTES];

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t first_dev = 0;
  vc4_deviceptr_t second_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdw_active_cols_zero_with_prior_dma_probe_ssavc4 program create failed");
  if (vc4Malloc(program, &first_dev, sizeof(first_words)) < 0 ||
      vc4Malloc(program, &second_dev, sizeof(second_bytes)) < 0)
    panic("vdw_active_cols_zero_with_prior_dma_probe_ssavc4 allocation failed");

  for (uint32_t i = 0; i < FIRST_WORDS + GUARD_WORDS; ++i)
    first_words[i] = SENTINEL_WORD;
  for (uint32_t i = 0; i < sizeof(second_bytes); ++i)
    second_bytes[i] = SENTINEL_BYTE;

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  if (vc4MemcpyHtoD(program, first_dev, first_words, sizeof(first_words)) < 0 ||
      vc4MemcpyHtoD(program, second_dev, second_bytes,
                    sizeof(second_bytes)) < 0 ||
      vdw_active_cols_zero_with_prior_dma_probe_ssavc4_launch(
          program, grid, block, first_dev, second_dev, 7u, 0u, 64u) < 0 ||
      vc4MemcpyDtoH(program, first_words, first_dev, sizeof(first_words)) < 0 ||
      vc4MemcpyDtoH(program, second_bytes, second_dev,
                    sizeof(second_bytes)) < 0) {
    printk("ERROR: vdw_active_cols_zero_with_prior_dma_probe launch/copy failed\n");
    ++launch_failures;
  }

  uint32_t expected_first = 0x80000000u;
  checksum += first_words[0];
  if (first_words[0] != expected_first) {
    printk("ERROR: vdw_active_cols_zero_with_prior_dma_probe first actual=%x expected=%x\n",
           first_words[0], expected_first);
    ++total_mismatches;
  }
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    uint32_t actual = first_words[FIRST_WORDS + i];
    if (actual != SENTINEL_WORD) {
      if (sentinel_mismatches < 8)
        printk("ERROR: vdw_active_cols_zero_with_prior_dma_probe first_guard=%d actual=%x expected=%x\n",
               (int)i, actual, SENTINEL_WORD);
      ++sentinel_mismatches;
    }
  }
  for (uint32_t i = 0; i < sizeof(second_bytes); ++i) {
    uint8_t actual = second_bytes[i];
    checksum += actual;
    if (actual != SENTINEL_BYTE) {
      if (sentinel_mismatches < 8)
        printk("ERROR: vdw_active_cols_zero_with_prior_dma_probe second_byte=%d actual=%x expected=%x\n",
               (int)i, actual, SENTINEL_BYTE);
      ++sentinel_mismatches;
    }
  }

  uint32_t runtime_launches =
      vdw_active_cols_zero_with_prior_dma_probe_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      vdw_active_cols_zero_with_prior_dma_probe_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_active_cols_zero_with_prior_dma_probe_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, first_dev);
  vc4Free(program, second_dev);
}
