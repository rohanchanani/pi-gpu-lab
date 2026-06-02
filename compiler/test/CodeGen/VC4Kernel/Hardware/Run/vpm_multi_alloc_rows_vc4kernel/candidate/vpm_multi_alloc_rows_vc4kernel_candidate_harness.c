#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ROWS 3u
#define WORDS (ROWS * LANES)
#define GUARD_WORDS 16u
#define BUFFER_WORDS (WORDS + GUARD_WORDS)
#define SENTINEL 0xdeadbeefu

static const uint32_t bases[ROWS] = {
    0x75000000u,
    0x76000000u,
    0x77000000u,
};

static uint32_t out_values[BUFFER_WORDS];

static void fill_output(void) {
  for (uint32_t i = 0; i < BUFFER_WORDS; ++i)
    out_values[i] = SENTINEL;
}

static int verify_active(uint32_t *checksum) {
  int mismatches = 0;
  *checksum = 0;
  for (uint32_t row = 0; row < ROWS; ++row) {
    for (uint32_t lane = 0; lane < LANES; ++lane) {
      uint32_t index = row * LANES + lane;
      uint32_t expected = bases[row] + lane;
      *checksum += out_values[index];
      if (out_values[index] != expected) {
        if (mismatches < 8)
          printk("ERROR: vpm_multi_alloc row=%d lane=%d actual=%x expected=%x\n",
                 (int)row, (int)lane, out_values[index], expected);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = WORDS; i < BUFFER_WORDS; ++i) {
    if (out_values[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: vpm_multi_alloc sentinel=%d actual=%x expected=%x\n",
               (int)i, out_values[i], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vpm_multi_alloc_rows_vc4kernel program create failed");

  vc4_deviceptr_t out_dev = 0;
  uint32_t bytes = BUFFER_WORDS * sizeof(uint32_t);
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("vpm_multi_alloc_rows_vc4kernel allocation failed");

  fill_output();
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
      vpm_multi_alloc_rows_vc4kernel_launch(program, grid, block, out_dev) <
          0 ||
      vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
    printk("ERROR: vpm_multi_alloc_rows_vc4kernel launch/copy failed\n");
    ++launch_failures;
  } else {
    total_mismatches = verify_active(&checksum);
    sentinel_mismatches = verify_sentinels();
  }

  launch_failures +=
      (int)vpm_multi_alloc_rows_vc4kernel_runtime_launch_failures();
  uint32_t runtime_launches = vpm_multi_alloc_rows_vc4kernel_runtime_launches();
  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vpm_multi_alloc_rows_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d rows=%d checksum_accum=%u runtime_allocations=1 runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)LANES, (int)ROWS, checksum, runtime_launches, elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
