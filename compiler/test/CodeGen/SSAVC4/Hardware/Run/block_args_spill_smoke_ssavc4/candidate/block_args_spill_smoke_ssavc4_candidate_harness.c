#include "rpi.h"
#include "vc4_m2_candidate_test_helpers.h"

#define BLOCK_ARGS_SPILL_SMOKE_SSAVC4_MAX_N 192u
#define BLOCK_ARGS_SPILL_SMOKE_SSAVC4_GUARD 32u
#define BLOCK_ARGS_SPILL_SMOKE_SSAVC4_BUFFER_N \
  (BLOCK_ARGS_SPILL_SMOKE_SSAVC4_MAX_N + BLOCK_ARGS_SPILL_SMOKE_SSAVC4_GUARD)
#define BLOCK_ARGS_SPILL_SMOKE_SSAVC4_SENTINEL 0xdeadbeefu
#define BLOCK_ARGS_SPILL_SMOKE_SSAVC4_ACTIVE_QPUS 12u
#define BLOCK_ARGS_SPILL_SMOKE_SSAVC4_LANE_WIDTH 16u
#define BLOCK_ARGS_SPILL_SMOKE_SSAVC4_ITERS 5u

static uint32_t out_values[BLOCK_ARGS_SPILL_SMOKE_SSAVC4_BUFFER_N];
static const uint32_t test_ns[] = {0u, 1u, 17u, 65u, 192u};

static uint32_t expected_value(uint32_t case_id, uint32_t index) {
  uint32_t lane = index & (BLOCK_ARGS_SPILL_SMOKE_SSAVC4_LANE_WIDTH - 1u);
  uint32_t acc = index + ((case_id & 0xffu) << 8);
  uint32_t case_step = (case_id & 0xffu) << 2;
  uint32_t carry = index + 1u;

  for (uint32_t iter = 0; iter < BLOCK_ARGS_SPILL_SMOKE_SSAVC4_ITERS; ++iter) {
    acc += lane + iter + case_step + carry;
    carry += 1u;
  }

  acc += carry;
  for (uint32_t i = 1; i <= 8u; ++i)
    acc += index + i;
  return acc;
}

static void fill_host_buffer(void) {
  for (uint32_t i = 0; i < BLOCK_ARGS_SPILL_SMOKE_SSAVC4_BUFFER_N; ++i)
    out_values[i] = BLOCK_ARGS_SPILL_SMOKE_SSAVC4_SENTINEL;
}

static int verify_results(uint32_t case_id, uint32_t n) {
  int mismatches = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t expected = expected_value(case_id, i);
    if (out_values[i] != expected) {
      if (mismatches < 8)
        printk("ERROR: block_args_spill_smoke_ssavc4 case=%d i=%d gpu=%x expected=%x\n",
               (int)case_id, (int)i, out_values[i], expected);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_sentinel_tail(uint32_t n) {
  int mismatches = 0;
  for (uint32_t i = n;
       i < n + BLOCK_ARGS_SPILL_SMOKE_SSAVC4_GUARD &&
       i < BLOCK_ARGS_SPILL_SMOKE_SSAVC4_BUFFER_N;
       ++i) {
    if (out_values[i] != BLOCK_ARGS_SPILL_SMOKE_SSAVC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: block_args_spill_smoke_ssavc4 sentinel changed i=%d value=%x expected=%x\n",
               (int)i, out_values[i], BLOCK_ARGS_SPILL_SMOKE_SSAVC4_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

static int checksum_low16(const uint32_t *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] & 0xffffu);
  return checksum;
}

void notmain(void) {
  struct vc4_program *program = 0;
  const uint32_t active_qpus = BLOCK_ARGS_SPILL_SMOKE_SSAVC4_ACTIVE_QPUS;
  const uint32_t lane_width = BLOCK_ARGS_SPILL_SMOKE_SSAVC4_LANE_WIDTH;
  const uint32_t case_count = sizeof(test_ns) / sizeof(test_ns[0]);
  const uint32_t bytes = BLOCK_ARGS_SPILL_SMOKE_SSAVC4_BUFFER_N * sizeof(uint32_t);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vc4_program_create failed");

  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("block_args_spill_smoke_ssavc4 device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(active_qpus * lane_width, 1, 1);

  printk("Running VC4 block_args_spill_smoke_ssavc4 candidate bundle...\n");
  printk("BLOCK_ARGS_SPILL_SMOKE_SSAVC4_SETUP max_n=%d active_qpus=%d lanes=%d iters=%d allocations=%d cases=%d\n",
         (int)BLOCK_ARGS_SPILL_SMOKE_SSAVC4_MAX_N, (int)active_qpus,
         (int)lane_width, (int)BLOCK_ARGS_SPILL_SMOKE_SSAVC4_ITERS, 1,
         (int)case_count);

  for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
    uint32_t n = test_ns[case_index];
    uint32_t case_id = case_index + 1u;
    fill_host_buffer();

    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        block_args_spill_smoke_ssavc4_launch(program, grid, block,
                                             out_dev, n, case_id) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
      printk("ERROR: block_args_spill_smoke_ssavc4 launch/copy failed case=%d n=%d\n",
             (int)case_id, (int)n);
      ++launch_failures;
      continue;
    }

    int mismatches = verify_results(case_id, n);
    int case_sentinel_mismatches = verify_sentinel_tail(n);
    int checksum = checksum_low16(out_values, n);
    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinel_mismatches;
    checksum_accum += checksum;
    printk("BLOCK_ARGS_SPILL_SMOKE_SSAVC4_CASE case=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
           (int)case_id, (int)n, mismatches, case_sentinel_mismatches, checksum);
  }

  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=block_args_spill_smoke_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, (int)active_qpus, (int)lane_width,
         (int)BLOCK_ARGS_SPILL_SMOKE_SSAVC4_MAX_N, checksum_accum,
         1, (int)case_count, elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
