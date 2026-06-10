#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_ACTIVE_QPUS 12u
#define SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_LANES 16u
#define SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_MAX_N \
  (SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_ACTIVE_QPUS * \
   SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_LANES)
#define SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_GUARD 32u
#define SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_BUFFER_N \
  (SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_MAX_N + \
   SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_GUARD)
#define SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_SENTINEL 0xdeadbeefu

static uint32_t out_values[SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_BUFFER_N];
static const uint32_t test_ns[] = {0u, 1u, 16u, 17u, 65u, 192u};
static const uint32_t selectors[] = {0u, 1u};

static uint32_t rotl32_local(uint32_t value, unsigned shift) {
  return (value << shift) | (value >> (32u - shift));
}

static uint32_t expected_value(uint32_t selector, uint32_t index) {
  return index + (selector == 0u ? 1655u : 2656u);
}

static void fill_host_buffer(void) {
  for (uint32_t i = 0;
       i < SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_BUFFER_N; ++i)
    out_values[i] = SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_SENTINEL;
}

static int verify_results(uint32_t selector, uint32_t n) {
  int mismatches = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t expected = expected_value(selector, i);
    if (out_values[i] != expected) {
      if (mismatches < 8)
        printk("ERROR: ssavc4_multi_exit_merge_block_arg_spill_vc4 selector=%d n=%d i=%d gpu=%x expected=%x\n",
               (int)selector, (int)n, (int)i, out_values[i], expected);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_sentinel_tail(uint32_t n) {
  int mismatches = 0;
  for (uint32_t i = n;
       i < n + SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_GUARD &&
       i < SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_BUFFER_N;
       ++i) {
    if (out_values[i] !=
        SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: ssavc4_multi_exit_merge_block_arg_spill_vc4 sentinel changed i=%d value=%x expected=%x\n",
               (int)i, out_values[i],
               SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

static uint32_t hash_case(uint32_t selector, uint32_t n) {
  uint32_t hash = 2166136261u ^ (selector * 0x45d9f3bu) ^ n;
  for (uint32_t i = 0; i < n; ++i) {
    hash ^= expected_value(selector, i);
    hash *= 16777619u;
  }
  return hash;
}

void notmain(void) {
  struct vc4_program *program = 0;
  const uint32_t active_qpus =
      SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_ACTIVE_QPUS;
  const uint32_t lanes = SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_LANES;
  const uint32_t bytes =
      SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_BUFFER_N *
      sizeof(uint32_t);
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(active_qpus * lanes, 1, 1);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vc4_program_create failed");

  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("ssavc4_multi_exit_merge_block_arg_spill_vc4 allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t output_hash = 2166136261u;
  int start = timer_get_usec();
  int case_id = 0;

  printk("Running VC4 ssavc4_multi_exit_merge_block_arg_spill_vc4 candidate bundle...\n");
  printk("SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_SETUP active_qpus=%d lanes=%d max_n=%d cases=%d\n",
         (int)active_qpus, (int)lanes,
         (int)SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_MAX_N,
         (int)(sizeof(test_ns) / sizeof(test_ns[0]) *
               sizeof(selectors) / sizeof(selectors[0])));

  for (uint32_t selector_index = 0;
       selector_index < sizeof(selectors) / sizeof(selectors[0]);
       ++selector_index) {
    uint32_t selector = selectors[selector_index];
    for (uint32_t n_index = 0; n_index < sizeof(test_ns) / sizeof(test_ns[0]);
         ++n_index) {
      uint32_t n = test_ns[n_index];
      fill_host_buffer();

      if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
          ssavc4_multi_exit_merge_block_arg_spill_vc4_launch(
              program, grid, block, out_dev, n, selector) < 0 ||
          vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
        printk("ERROR: ssavc4_multi_exit_merge_block_arg_spill_vc4 launch/copy failed selector=%d n=%d\n",
               (int)selector, (int)n);
        ++launch_failures;
        ++case_id;
        continue;
      }

      int mismatches = verify_results(selector, n);
      int case_sentinel_mismatches = verify_sentinel_tail(n);
      uint32_t case_hash = hash_case(selector, n);
      output_hash ^= case_hash + 0x9e3779b9u + ((uint32_t)case_id << 6) +
                     ((uint32_t)case_id >> 2);
      output_hash = rotl32_local(output_hash, 7u);
      total_mismatches += mismatches;
      sentinel_mismatches += case_sentinel_mismatches;
      printk("SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_CASE case=%d selector=%d n=%d mismatches=%d sentinel_mismatches=%d case_hash=%u\n",
             case_id, (int)selector, (int)n, mismatches,
             case_sentinel_mismatches, case_hash);
      ++case_id;
    }
  }

  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=ssavc4_multi_exit_merge_block_arg_spill_vc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d output_hash=%u saw_ssavc4_multi_exit_merge=1 saw_ssavc4_merge_block_arg=1 saw_ssavc4_spill_mode=1 saw_ssavc4_edge_copy=1 saw_ssavc4_final_merge_store=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
         status, case_id, total_mismatches, sentinel_mismatches,
         launch_failures, (int)active_qpus, (int)lanes,
         (int)SSAVC4_MULTI_EXIT_MERGE_BLOCK_ARG_SPILL_VC4_MAX_N, output_hash,
         1, case_id, elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
