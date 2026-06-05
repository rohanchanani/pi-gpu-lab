#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_EPSILON 0.0002f
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_CHECKSUM_SCALE 4096.0f
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS 12u
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_WAVES 1u
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES 16u
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_GUARD 32u
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_SENTINEL (-34567.0f)
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_ID_SENTINEL 0xffffffffu
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_K 17u
#define LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_BETA 0.03125f

static float out_values[LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES + LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_GUARD];
static float expected_values[LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES + LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_GUARD];
static uint32_t id_values[LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES + LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_GUARD];
static uint32_t expected_ids[LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES + LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_GUARD];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static void fill_buffers(void) {
  for (uint32_t i = 0; i < LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES + LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_GUARD; ++i) {
    out_values[i] = LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_SENTINEL;
    expected_values[i] = LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_SENTINEL;
    id_values[i] = LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_ID_SENTINEL;
    expected_ids[i] = LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_ID_SENTINEL;
  }
}

static void run_cpu_reference(void) {
  float value = ((2.0f * (float)LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_K) + 322.0f) * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_BETA;
  for (uint32_t req = 0; req < LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS; ++req) {
    for (uint32_t lane = 0; lane < LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES; ++lane) {
      uint32_t idx = req * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES + lane;
      expected_values[idx] = value;
      expected_ids[idx] = idx;
    }
  }
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_CHECKSUM_SCALE);
  return checksum;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t out_dev = 0, ids_dev = 0;
  const uint32_t active_count = LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS * LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES;
  const uint32_t total_count = active_count + LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_GUARD;
  const uint32_t out_bytes = total_count * sizeof(float);
  const uint32_t id_bytes = total_count * sizeof(uint32_t);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("large_spill_frame_multi_resident_ssavc4 program create failed");
  if (vc4_m2_malloc(program, &out_dev, out_bytes) < 0 ||
      vc4_m2_malloc(program, &ids_dev, id_bytes) < 0)
    panic("large_spill_frame_multi_resident_ssavc4 allocation failed");

  fill_buffers();
  run_cpu_reference();
  int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
  float max_abs_diff = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(active_count, 1, 1);

  if (vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
      vc4_m2_copy_htod(program, ids_dev, id_values, id_bytes) < 0 ||
      large_spill_frame_multi_resident_ssavc4_launch(program, grid, block, out_dev, ids_dev, LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_K, LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_BETA) < 0 ||
      vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0 ||
      vc4_m2_copy_dtoh(program, id_values, ids_dev, id_bytes) < 0) {
    printk("ERROR: large_spill_frame_multi_resident_ssavc4 launch/copy failed\n");
    ++launch_failures;
  } else {
    for (uint32_t i = 0; i < active_count; ++i) {
      float diff = out_values[i] - expected_values[i];
      float abs_diff = absf_local(diff);
      if (abs_diff > max_abs_diff)
        max_abs_diff = abs_diff;
      if (abs_diff > LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_EPSILON) {
        if (total_mismatches < 8)
          printk("ERROR: large_spill_frame_multi_resident_ssavc4 value i=%d gpu=%f expected=%f diff=%f\n", (int)i, out_values[i], expected_values[i], diff);
        ++total_mismatches;
      }
      if (id_values[i] != expected_ids[i]) {
        if (total_mismatches < 8)
          printk("ERROR: large_spill_frame_multi_resident_ssavc4 id i=%d gpu=%x expected=%x\n", (int)i, id_values[i], expected_ids[i]);
        ++total_mismatches;
      }
    }
    for (uint32_t i = active_count; i < total_count; ++i) {
      if (out_values[i] != LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_SENTINEL || id_values[i] != LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_ID_SENTINEL)
        ++sentinel_mismatches;
    }
  }

  int checksum = scaled_checksum(out_values, active_count);
  int expected_checksum = scaled_checksum(expected_values, active_count);
  if (checksum != expected_checksum) {
    printk("ERROR: large_spill_frame_multi_resident_ssavc4 checksum mismatch gpu=%d expected=%d\n", checksum, expected_checksum);
    ++total_mismatches;
  }
  launch_failures += (int)large_spill_frame_multi_resident_ssavc4_runtime_launch_failures();
  uint32_t launches = large_spill_frame_multi_resident_ssavc4_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0 && launches == 1) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=large_spill_frame_multi_resident_ssavc4 status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d requests=%d waves=%d lanes=%d forced_spill_terms=24 checksum=%d max_abs_diff=%f runtime_allocations=2 runtime_launches=%d elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_REQUESTS, (int)LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_WAVES, (int)LARGE_SPILL_FRAME_MULTI_RESIDENT_SSAVC4_LANES,
         checksum, max_abs_diff, launches, timer_get_usec() - start);

  vc4Free(program, out_dev);
  vc4Free(program, ids_dev);
  vc4_program_destroy(program);
}
