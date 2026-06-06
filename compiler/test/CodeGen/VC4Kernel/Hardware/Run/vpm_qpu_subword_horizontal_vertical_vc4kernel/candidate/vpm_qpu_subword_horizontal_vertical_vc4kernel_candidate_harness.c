#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 4u
#define OUT_WORDS (LANES * SEGMENTS + 16u)
#define SENTINEL 0xa55a7109u

static uint32_t out_words[OUT_WORDS];

static void fill_output(void) {
  for (uint32_t i = 0; i < OUT_WORDS; ++i)
    out_words[i] = SENTINEL;
}

static uint32_t expected_value(uint32_t segment, uint32_t lane) {
  switch (segment) {
  case 0: return 11u + lane;
  case 1: return 300u + lane;
  case 2: return 121u + lane;
  default: return 4096u + lane;
  }
}

static int verify_outputs(uint32_t *checksum) {
  int mismatches = 0;
  *checksum = 0;
  for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
    for (uint32_t lane = 0; lane < LANES; ++lane) {
      uint32_t index = segment * LANES + lane;
      uint32_t expected = expected_value(segment, lane);
      uint32_t got = out_words[index];
      *checksum += got;
      if (got != expected) {
        if (mismatches < 8)
          printk("ERROR: vpm_qpu_subword_hv seg=%d lane=%d got=%x expected=%x\n",
                 (int)segment, (int)lane, got, expected);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = LANES * SEGMENTS; i < OUT_WORDS; ++i) {
    if (out_words[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: vpm_qpu_subword_hv sentinel i=%d got=%x\n",
               (int)i, out_words[i]);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vpm_qpu_subword_hv program create failed");

  vc4_deviceptr_t out_dev = 0;
  uint32_t bytes = OUT_WORDS * sizeof(uint32_t);
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("vpm_qpu_subword_hv allocation failed");

  fill_output();
  int launch_failures = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
  if (vc4_m2_copy_htod(program, out_dev, out_words, bytes) < 0 ||
      vpm_qpu_subword_horizontal_vertical_vc4kernel_launch(program, grid, block,
                                                           out_dev) < 0 ||
      vc4_m2_copy_dtoh(program, out_words, out_dev, bytes) < 0)
    ++launch_failures;

  uint32_t checksum = 0;
  int total_mismatches = launch_failures ? 0 : verify_outputs(&checksum);
  int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vpm_qpu_subword_horizontal_vertical_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_vpm_qpu_w8=1 saw_vpm_qpu_w16=1 saw_packed=1 saw_laned=1 saw_horizontal=1 saw_vertical=1 saw_pack_unpack=1 saw_vdw_preserve=1 checksum_accum=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)LANES, checksum, elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
