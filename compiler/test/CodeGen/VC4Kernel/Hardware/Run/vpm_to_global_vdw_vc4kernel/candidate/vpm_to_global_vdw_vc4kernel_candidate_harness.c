#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ROWS 3u
#define COLS 16u
#define WORDS (ROWS * COLS)
#define GUARD 32u
#define SENTINEL 0x5a5a5a5au
#define BASE0 0x52000000u
#define BASE1 0x53000000u
#define TAIL_N 8u

static uint32_t output_values[WORDS + GUARD];

static void fill_output(void) {
  for (uint32_t i = 0; i < WORDS + GUARD; ++i)
    output_values[i] = SENTINEL;
}

static uint32_t expected(uint32_t row, uint32_t col) {
  if (row == 0u)
    return BASE0 + col;
  if (row == 1u && col < TAIL_N)
    return BASE1 + col;
  return SENTINEL;
}

static int verify(uint32_t *checksum) {
  int mismatches = 0;
  *checksum = 0;
  for (uint32_t r = 0; r < ROWS; ++r) {
    for (uint32_t c = 0; c < COLS; ++c) {
      uint32_t got = output_values[r * COLS + c];
      uint32_t want = expected(r, c);
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: vpm_to_vdw row=%d col=%d got=%x want=%x\n",
                 (int)r, (int)c, got, want);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_guard(void) {
  int mismatches = 0;
  for (uint32_t i = WORDS; i < WORDS + GUARD; ++i)
    if (output_values[i] != SENTINEL)
      ++mismatches;
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vpm_to_global_vdw_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &out_dev, (WORDS + GUARD) * sizeof(uint32_t)) < 0)
    panic("vpm_to_global_vdw_vc4kernel allocation failed");
  fill_output();
  int launch_failures = 0, total_mismatches = 0, sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  if (vc4_m2_copy_htod(program, out_dev, output_values, (WORDS + GUARD) * sizeof(uint32_t)) < 0 ||
      vpm_to_global_vdw_vc4kernel_launch(program, grid, block, out_dev, TAIL_N) < 0 ||
      vc4_m2_copy_dtoh(program, output_values, out_dev, (WORDS + GUARD) * sizeof(uint32_t)) < 0) {
    ++launch_failures;
  } else {
    total_mismatches = verify(&checksum);
    sentinel_mismatches = verify_guard();
  }
  launch_failures += (int)vpm_to_global_vdw_vc4kernel_runtime_launch_failures();
  uint32_t launches = vpm_to_global_vdw_vc4kernel_runtime_launches();
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vpm_to_global_vdw_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d rows=%d cols=%d checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)ROWS, (int)COLS, checksum, launches, timer_get_usec() - start);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
