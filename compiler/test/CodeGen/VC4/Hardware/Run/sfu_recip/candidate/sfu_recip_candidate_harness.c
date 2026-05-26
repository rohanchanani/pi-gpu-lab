#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SFU_RECIP_EPSILON 0.0003f
#define CHECKSUM_SCALE 1024.0f
#define SFU_RECIP_ACTIVE_QPUS 12u
#define SFU_RECIP_LANE_WIDTH 16u
#define SFU_RECIP_N (SFU_RECIP_ACTIVE_QPUS * SFU_RECIP_LANE_WIDTH)
#define SFU_RECIP_GUARD_WORDS 16u
#define SFU_RECIP_SENTINEL (-99.0f)

static float x_values[SFU_RECIP_N];
static float out_values[SFU_RECIP_N + SFU_RECIP_GUARD_WORDS];
static float expected_values[SFU_RECIP_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float pattern_value(uint32_t index) {
    switch (index & 7u) {
    case 0: return 0.5f;
    case 1: return 1.0f;
    case 2: return 2.0f;
    case 3: return 4.0f;
    case 4: return 8.0f;
    case 5: return 16.0f;
    case 6: return 0.25f;
    default: return 32.0f;
    }
}

static void fill_inputs(uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        x_values[i] = pattern_value(i);
        out_values[i] = SFU_RECIP_SENTINEL;
        expected_values[i] = 1.0f / x_values[i];
    }
    for (uint32_t i = n; i < n + SFU_RECIP_GUARD_WORDS; ++i)
        out_values[i] = SFU_RECIP_SENTINEL;
}

static int scaled_checksum(const float *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; ++i)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t n, int *mismatches, float *max_abs_diff) {
    *mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float diff = out_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;
        if (abs_diff > SFU_RECIP_EPSILON) {
            if (*mismatches < 8)
                printk("ERROR: i=%u x=%f gpu=%f cpu=%f diff=%f\n",
                       i, x_values[i], out_values[i], expected_values[i], diff);
            (*mismatches)++;
        }
    }
}

static int verify_guard(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < n + SFU_RECIP_GUARD_WORDS; ++i) {
        if (out_values[i] != SFU_RECIP_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: guard changed i=%u gpu=%f expected=%f\n",
                       i, out_values[i], SFU_RECIP_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    fill_inputs(SFU_RECIP_N);
    vc4_deviceptr_t x_dev = 0, out_dev = 0;
    uint32_t x_bytes = SFU_RECIP_N * sizeof(float);
    uint32_t out_bytes = (SFU_RECIP_N + SFU_RECIP_GUARD_WORDS) * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0 ||
        vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0)
        panic("sfu_recip device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(SFU_RECIP_N, 1, 1);

    printk("Running VC4 sfu_recip M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launch_failures = 0;
    if (sfu_recip_launch(program, grid, block, x_dev, out_dev, SFU_RECIP_N) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0)
        launch_failures++;
    int elapsed = timer_get_usec() - start;

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(SFU_RECIP_N, &mismatches, &max_abs_diff);
    int sentinel_mismatches = verify_guard(SFU_RECIP_N);
    int checksum = scaled_checksum(out_values, SFU_RECIP_N);
    uint32_t runtime_allocations = sfu_recip_runtime_allocations();
    uint32_t runtime_launches = sfu_recip_runtime_launches();

    printk("VC4_TEST_RESULT name=sfu_recip status=%s checked_elements=%u mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%u lanes=%u n=%u checksum=%d max_abs_diff=%f runtime_allocations=%u runtime_launches=%u elapsed_usec=%d\n",
           (mismatches || sentinel_mismatches || launch_failures) ? "FAIL" : "PASS",
           SFU_RECIP_N, mismatches, sentinel_mismatches, launch_failures,
           SFU_RECIP_ACTIVE_QPUS, SFU_RECIP_LANE_WIDTH, SFU_RECIP_N, checksum,
           max_abs_diff, runtime_allocations, runtime_launches, elapsed);
    if (mismatches || sentinel_mismatches)
        panic("sfu_recip verification failed: mismatches=%d sentinel_mismatches=%d max_abs_diff=%f",
              mismatches, sentinel_mismatches, max_abs_diff);

    vc4Free(program, x_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
