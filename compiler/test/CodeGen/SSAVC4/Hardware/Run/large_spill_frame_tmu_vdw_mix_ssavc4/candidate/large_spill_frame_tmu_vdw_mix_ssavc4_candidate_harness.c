#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_EPSILON 0.0002f
#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_CHECKSUM_SCALE 4096.0f
#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS 12u
#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_WAVES 1u
#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES 16u
#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_GUARD 32u
#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_SENTINEL (-45678.0f)
#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_ID_SENTINEL 0xffffffffu
#define LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_BETA 0.03125f

static const uint32_t k_cases[] = {0u, 2u, 5u};
static float x_values[LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES];
static float out_values[LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES + LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_GUARD];
static float expected_values[LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES + LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_GUARD];
static uint32_t id_values[LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES + LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_GUARD];
static uint32_t expected_ids[LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES + LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_GUARD];

static float absf_local(float value) { return value < 0.0f ? -value : value; }
static float x_value(uint32_t req, uint32_t lane) { return ((float)((req * 17u + lane * 3u + 5u) % 31u) - 15.0f) * 0.0625f; }

static void fill_inputs(void) {
  for (uint32_t req = 0; req < LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS; ++req)
    for (uint32_t lane = 0; lane < LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES; ++lane)
      x_values[req * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES + lane] = x_value(req, lane);
}

static void fill_buffers(void) {
  for (uint32_t i = 0; i < LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES + LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_GUARD; ++i) {
    out_values[i] = LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_SENTINEL;
    expected_values[i] = LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_SENTINEL;
    id_values[i] = LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_ID_SENTINEL;
    expected_ids[i] = LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_ID_SENTINEL;
  }
}

static void run_cpu_reference(uint32_t k) {
  for (uint32_t req = 0; req < LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS; ++req) {
    for (uint32_t lane = 0; lane < LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES; ++lane) {
      uint32_t idx = req * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES + lane;
      float x = x_value(req, lane);
      expected_values[idx] = ((float)k + 24.0f) * x + 300.0f * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_BETA;
      expected_ids[idx] = idx;
    }
  }
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_CHECKSUM_SCALE);
  return checksum;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t x_dev = 0, out_dev = 0, ids_dev = 0;
  const uint32_t case_count = sizeof(k_cases) / sizeof(k_cases[0]);
  const uint32_t active_count = LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES;
  const uint32_t total_count = active_count + LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_GUARD;
  const uint32_t out_bytes = total_count * sizeof(float);
  const uint32_t id_bytes = total_count * sizeof(uint32_t);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("large_spill_frame_tmu_vdw_mix_ssavc4 program create failed");
  if (vc4_m2_malloc(program, &x_dev, sizeof(x_values)) < 0 ||
      vc4_m2_malloc(program, &out_dev, out_bytes) < 0 ||
      vc4_m2_malloc(program, &ids_dev, id_bytes) < 0)
    panic("large_spill_frame_tmu_vdw_mix_ssavc4 allocation failed");

  fill_inputs();
  int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
  int checksum_accum = 0;
  float max_abs_diff_overall = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(active_count, 1, 1);

  for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
    uint32_t k = k_cases[case_index];
    fill_buffers();
    run_cpu_reference(k);
    if (vc4_m2_copy_htod(program, x_dev, x_values, sizeof(x_values)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
        vc4_m2_copy_htod(program, ids_dev, id_values, id_bytes) < 0 ||
        large_spill_frame_tmu_vdw_mix_ssavc4_launch(program, grid, block, x_dev, out_dev, ids_dev, k, LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_BETA) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0 ||
        vc4_m2_copy_dtoh(program, id_values, ids_dev, id_bytes) < 0) {
      printk("ERROR: large_spill_frame_tmu_vdw_mix_ssavc4 launch/copy failed case=%d k=%d\n", (int)case_index, (int)k);
      ++launch_failures;
      continue;
    }
    int case_mismatches = 0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < active_count; ++i) {
      float diff = out_values[i] - expected_values[i];
      float abs_diff = absf_local(diff);
      if (abs_diff > max_abs_diff)
        max_abs_diff = abs_diff;
      if (abs_diff > LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_EPSILON) {
        if (case_mismatches < 8)
          printk("ERROR: large_spill_frame_tmu_vdw_mix_ssavc4 value case=%d i=%d gpu=%f expected=%f diff=%f\n", (int)case_index, (int)i, out_values[i], expected_values[i], diff);
        ++case_mismatches;
      }
      if (id_values[i] != expected_ids[i]) {
        if (case_mismatches < 8)
          printk("ERROR: large_spill_frame_tmu_vdw_mix_ssavc4 id case=%d i=%d gpu=%x expected=%x\n", (int)case_index, (int)i, id_values[i], expected_ids[i]);
        ++case_mismatches;
      }
    }
    for (uint32_t i = active_count; i < total_count; ++i) {
      if (out_values[i] != LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_SENTINEL || id_values[i] != LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_ID_SENTINEL)
        ++sentinel_mismatches;
    }
    int checksum = scaled_checksum(out_values, active_count);
    int expected_checksum = scaled_checksum(expected_values, active_count);
    if (checksum != expected_checksum) {
      printk("ERROR: large_spill_frame_tmu_vdw_mix_ssavc4 checksum mismatch case=%d gpu=%d expected=%d\n", (int)case_index, checksum, expected_checksum);
      ++case_mismatches;
    }
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += case_mismatches;
    checksum_accum += checksum;
    printk("LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_CASE case=%d k=%d mismatches=%d checksum=%d max_abs_diff=%f\n", (int)case_index, (int)k, case_mismatches, checksum, max_abs_diff);
  }

  launch_failures += (int)large_spill_frame_tmu_vdw_mix_ssavc4_runtime_launch_failures();
  uint32_t launches = large_spill_frame_tmu_vdw_mix_ssavc4_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0 && launches == case_count) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=large_spill_frame_tmu_vdw_mix_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d requests=%d waves=%d lanes=%d forced_spill_terms=24 checksum_accum=%d max_abs_diff=%f runtime_allocations=3 runtime_launches=%d elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches, launch_failures,
         (int)LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_REQUESTS, (int)LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_WAVES, (int)LARGE_SPILL_FRAME_TMU_VDW_MIX_SSAVC4_LANES,
         checksum_accum, max_abs_diff_overall, launches, timer_get_usec() - start);

  vc4Free(program, x_dev);
  vc4Free(program, out_dev);
  vc4Free(program, ids_dev);
  vc4_program_destroy(program);
}
