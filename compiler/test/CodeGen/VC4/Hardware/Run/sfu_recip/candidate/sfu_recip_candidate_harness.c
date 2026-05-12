#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SFU_RECIP_EPSILON 0.0003f
#define CHECKSUM_SCALE 1024.0f
#define SFU_RECIP_ACTIVE_QPUS 12u
#define SFU_RECIP_LANE_WIDTH 16u
#define SFU_RECIP_N (SFU_RECIP_ACTIVE_QPUS * SFU_RECIP_LANE_WIDTH)

static float x_values[SFU_RECIP_N];
static float out_values[SFU_RECIP_N];
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
        out_values[i] = -99.0f;
        expected_values[i] = 1.0f / x_values[i];
    }
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

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    fill_inputs(SFU_RECIP_N);
    vc4_deviceptr_t x_dev = 0, out_dev = 0;
    uint32_t bytes = SFU_RECIP_N * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0 ||
        vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0)
        panic("sfu_recip device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(SFU_RECIP_N, 1, 1);

    printk("Running VC4 sfu_recip M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launch_failures = 0;
    if (sfu_recip_launch(program, grid, block, x_dev, out_dev, SFU_RECIP_N) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0)
        launch_failures++;
    int elapsed = timer_get_usec() - start;

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(SFU_RECIP_N, &mismatches, &max_abs_diff);
    int checksum = scaled_checksum(out_values, SFU_RECIP_N);

    printk("VC4_TEST_RESULT name=sfu_recip status=%s mismatches=%d active_qpus=%u n=%u checksum=%d max_abs_diff=%f elapsed_usec=%d\n",
           (mismatches || launch_failures) ? "FAIL" : "PASS",
           mismatches, SFU_RECIP_ACTIVE_QPUS, SFU_RECIP_N, checksum, max_abs_diff, elapsed);
    if (mismatches)
        panic("sfu_recip verification failed: mismatches=%d max_abs_diff=%f", mismatches, max_abs_diff);

    vc4Free(program, x_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
