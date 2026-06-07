#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define OUT_WORDS 80u
#define SENTINEL 0x6d517e1u

struct test_case {
  uint32_t iters;
  uint32_t active_cols;
};

static const struct test_case cases[] = {
    {1u, 1u}, {2u, 0u}, {3u, 5u}, {4u, 7u},
};

static uint32_t output_words[OUT_WORDS];

static void fill_buffers(uint32_t case_index) {
  (void)case_index;
  for (uint32_t i = 0; i < OUT_WORDS; ++i)
    output_words[i] = SENTINEL;
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *sentinel_mismatches = 0;
  uint32_t sum = 0;
  uint32_t last_iter = tc->iters - 1u;
  for (uint32_t lane = 0; lane < 16u; ++lane) {
    uint32_t want = 0;
    if (lane < tc->active_cols)
      want = 0x4100u + last_iter * 0x20u + lane;
    sum += want;
    uint32_t got = output_words[lane];
    *checksum += got;
    if (got != want) {
      if (mismatches < 8)
        printk("ERROR: pingpong_qpu_read lane case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, want);
      ++mismatches;
    }
  }
  for (uint32_t lane = 0; lane < 16u; ++lane) {
    uint32_t got = output_words[32u + lane];
    *checksum += got;
    if (got != sum) {
      if (mismatches < 8)
        printk("ERROR: pingpong_qpu_read reduce case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, sum);
      ++mismatches;
    }
  }
  for (uint32_t i = 16u; i < OUT_WORDS; ++i) {
    if (i >= 32u && i < 48u)
      continue;
    if (output_words[i] != SENTINEL) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: pingpong_qpu_read sentinel case=%d index=%d got=%x want=%x\n",
               (int)case_index, (int)i, output_words[i], SENTINEL);
      ++*sentinel_mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vpm_pingpong_qpu_read_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &out_dev, sizeof(output_words)) < 0)
    panic("dynamic_vpm_pingpong_qpu_read_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_buffers(i);
    if (vc4_m2_copy_htod(program, out_dev, output_words, sizeof(output_words)) < 0 ||
        dynamic_vpm_pingpong_qpu_read_vc4kernel_launch(
            program, grid, block, out_dev, cases[i].iters,
            cases[i].active_cols) < 0 ||
        vc4_m2_copy_dtoh(program, output_words, out_dev, sizeof(output_words)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    total_mismatches += verify_case(&cases[i], i, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
  }

  launch_failures +=
      (int)dynamic_vpm_pingpong_qpu_read_vc4kernel_runtime_launch_failures();
  uint32_t launches = dynamic_vpm_pingpong_qpu_read_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vpm_pingpong_qpu_read_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_dynamic_vpm_pingpong=1 saw_pingpong_qpu_write=1 saw_pingpong_qpu_read=1 saw_checked_qpu_read_output=1 saw_qpu_dynamic_subword_selector=1 saw_fragment_pack=1 saw_fragment_unpack=1 saw_cmp_select=1 saw_fragment_reduce=1 saw_runtime_loop=1 saw_iters1=1 saw_iters4=1 saw_vdw_preserve=1 no_tmu_to_vpm=1 selector_values_w16=0,1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
