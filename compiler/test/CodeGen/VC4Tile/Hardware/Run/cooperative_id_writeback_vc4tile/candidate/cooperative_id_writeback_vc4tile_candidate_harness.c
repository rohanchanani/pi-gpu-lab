#include "vc4_m2_candidate_test_helpers.h"
#include "vc4_case_config.h"
#include "kernel_launch.h"

#include <stdint.h>

#define COOP_ID_WRITEBACK_BLOCKS 2u
#ifndef VC4_CASE_WARPS_PER_BLOCK
#define VC4_CASE_WARPS_PER_BLOCK 2
#endif
#define COOP_ID_WRITEBACK_WARPS_PER_BLOCK ((unsigned)VC4_CASE_WARPS_PER_BLOCK)
#define COOP_ID_WRITEBACK_MAX_WARPS_PER_BLOCK 12u
#define COOP_ID_WRITEBACK_LANES 16u
#define COOP_ID_WRITEBACK_LAYOUT_WORDS \
  (COOP_ID_WRITEBACK_BLOCKS * COOP_ID_WRITEBACK_MAX_WARPS_PER_BLOCK * \
   COOP_ID_WRITEBACK_LANES)
#define COOP_ID_WRITEBACK_SENTINEL 0xc0091d5u

static uint32_t out_values[COOP_ID_WRITEBACK_LAYOUT_WORDS + 16u];

static void fill_host_buffer(void) {
  for (unsigned i = 0; i < COOP_ID_WRITEBACK_LAYOUT_WORDS + 16u; ++i)
    out_values[i] = COOP_ID_WRITEBACK_SENTINEL;
}

static uint32_t expected_word(unsigned block, unsigned warp, unsigned lane) {
  return block * 1000u + warp * COOP_ID_WRITEBACK_LANES + lane;
}

static int is_active_word(unsigned index) {
  for (unsigned b = 0; b < COOP_ID_WRITEBACK_BLOCKS; ++b) {
    unsigned block_base =
        b * COOP_ID_WRITEBACK_MAX_WARPS_PER_BLOCK * COOP_ID_WRITEBACK_LANES;
    unsigned active_begin = block_base;
    unsigned active_end =
        block_base + COOP_ID_WRITEBACK_WARPS_PER_BLOCK * COOP_ID_WRITEBACK_LANES;
    if (index >= active_begin && index < active_end)
      return 1;
  }
  return 0;
}

void notmain(void) {
  if (COOP_ID_WRITEBACK_WARPS_PER_BLOCK == 0u ||
      COOP_ID_WRITEBACK_WARPS_PER_BLOCK > COOP_ID_WRITEBACK_MAX_WARPS_PER_BLOCK) {
    printk("VC4_TEST_RESULT name=cooperative_id_writeback_vc4tile status=FAIL cases=1 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 blocks=%d warps_per_block=%d lanes=%d active_words=%d layout_words=%d physical_warp_slots=%d reason=invalid_warps_per_block\n",
           (int)COOP_ID_WRITEBACK_BLOCKS,
           (int)COOP_ID_WRITEBACK_WARPS_PER_BLOCK,
           (int)COOP_ID_WRITEBACK_LANES,
           (int)(COOP_ID_WRITEBACK_BLOCKS * COOP_ID_WRITEBACK_WARPS_PER_BLOCK *
                 COOP_ID_WRITEBACK_LANES),
           (int)COOP_ID_WRITEBACK_LAYOUT_WORDS,
           (int)COOP_ID_WRITEBACK_MAX_WARPS_PER_BLOCK);
    return;
  }

  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("cooperative_id_writeback_vc4tile vc4_program_create failed");

  const uint32_t bytes = sizeof(out_values);
  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("cooperative_id_writeback_vc4tile device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;

  vc4_dim3 grid = vc4_m2_dim3(COOP_ID_WRITEBACK_BLOCKS, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(
      COOP_ID_WRITEBACK_WARPS_PER_BLOCK * COOP_ID_WRITEBACK_LANES, 1, 1);

  printk("Running VC4 cooperative_id_writeback_vc4tile candidate bundle...\n");
  fill_host_buffer();
  int start = timer_get_usec();
  if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
      cooperative_id_writeback_vc4tile_launch(program, grid, block, out_dev) < 0 ||
      vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
    printk("ERROR: cooperative_id_writeback_vc4tile launch/copy failed\n");
    launch_failures++;
  } else {
    for (unsigned b = 0; b < COOP_ID_WRITEBACK_BLOCKS; ++b) {
      for (unsigned w = 0; w < COOP_ID_WRITEBACK_WARPS_PER_BLOCK; ++w) {
        for (unsigned lane = 0; lane < COOP_ID_WRITEBACK_LANES; ++lane) {
          unsigned idx =
              ((b * COOP_ID_WRITEBACK_MAX_WARPS_PER_BLOCK) + w) *
                  COOP_ID_WRITEBACK_LANES +
              lane;
          uint32_t expected = expected_word(b, w, lane);
          checksum_accum += out_values[idx];
          if (out_values[idx] != expected) {
            if (total_mismatches < 12) {
              printk("MISMATCH cooperative_id_writeback_vc4tile block=%u warp=%u lane=%u observed=%x expected=%x\n",
                     b, w, lane, out_values[idx], expected);
            }
            total_mismatches++;
          }
        }
      }
    }
    for (unsigned i = 0; i < COOP_ID_WRITEBACK_LAYOUT_WORDS + 16u; ++i) {
      if (i < COOP_ID_WRITEBACK_LAYOUT_WORDS && is_active_word(i))
        continue;
      if (out_values[i] != COOP_ID_WRITEBACK_SENTINEL)
        sentinel_mismatches++;
    }
  }

  int elapsed = timer_get_usec() - start;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=cooperative_id_writeback_vc4tile status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d blocks=%d warps_per_block=%d lanes=%d active_words=%d layout_words=%d physical_warp_slots=%d total_requests=%d checksum_accum=%u runtime_allocations=%u runtime_launches=%u elapsed_usec=%d\n",
         status, 1, total_mismatches, sentinel_mismatches, launch_failures,
         (int)COOP_ID_WRITEBACK_BLOCKS, (int)COOP_ID_WRITEBACK_WARPS_PER_BLOCK,
         (int)COOP_ID_WRITEBACK_LANES,
         (int)(COOP_ID_WRITEBACK_BLOCKS * COOP_ID_WRITEBACK_WARPS_PER_BLOCK *
               COOP_ID_WRITEBACK_LANES),
         (int)COOP_ID_WRITEBACK_LAYOUT_WORDS,
         (int)COOP_ID_WRITEBACK_MAX_WARPS_PER_BLOCK,
         (int)(COOP_ID_WRITEBACK_BLOCKS * COOP_ID_WRITEBACK_WARPS_PER_BLOCK),
         checksum_accum, cooperative_id_writeback_vc4tile_runtime_allocations(),
         cooperative_id_writeback_vc4tile_runtime_launches(), elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
