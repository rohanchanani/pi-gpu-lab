#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SAXPY_TMU_EPSILON 0.0001f
#define CHECKSUM_SCALE 1024.0f
#define SAXPY_TMU_ACTIVE_QPUS 12u
#define SAXPY_TMU_LANE_WIDTH 16u
#define SAXPY_TMU_N (SAXPY_TMU_ACTIVE_QPUS * SAXPY_TMU_LANE_WIDTH)

static float x_values[SAXPY_TMU_N];
static float y_values[SAXPY_TMU_N];
static float y_initial[SAXPY_TMU_N];
static float expected_values[SAXPY_TMU_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static void fill_inputs(uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        x_values[i] = ((float)(i % 97u) * 0.125f) - 3.0f;
        y_values[i] = ((float)(i % 53u) * 0.25f) + 1.0f;
        y_initial[i] = y_values[i];
    }
}

static void run_cpu_reference(float alpha, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        expected_values[i] = alpha * x_values[i] + y_initial[i];
}

static int scaled_checksum(const float *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t n, int *mismatch_count, float *max_abs_diff) {
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float diff = y_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;
        if (abs_diff > SAXPY_TMU_EPSILON) {
            if (*mismatch_count < 8)
                printk("ERROR: i=%d gpu=%f cpu=%f diff=%f\n", (int)i, y_values[i], expected_values[i], diff);
            (*mismatch_count)++;
        }
    }
}

void notmain(void) {
    struct vc4_program *program = 0;
    const float alpha = 2.5f;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    fill_inputs(SAXPY_TMU_N);
    run_cpu_reference(alpha, SAXPY_TMU_N);

    vc4_deviceptr_t x_dev = 0, y_dev = 0;
    uint32_t bytes = SAXPY_TMU_N * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0 ||
        vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
        vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0)
        panic("saxpy_tmu device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(SAXPY_TMU_N, 1, 1);

    printk("Running VC4 saxpy_tmu M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launch_failures = 0;
    if (saxpy_tmu_launch(program, grid, block, x_dev, y_dev, alpha, SAXPY_TMU_N) < 0 ||
        vc4_m2_copy_dtoh(program, y_values, y_dev, bytes) < 0)
        launch_failures++;
    int elapsed = timer_get_usec() - start;

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(SAXPY_TMU_N, &mismatches, &max_abs_diff);
    int checksum = scaled_checksum(y_values, SAXPY_TMU_N);
    int expected_checksum = scaled_checksum(expected_values, SAXPY_TMU_N);
    if (checksum != expected_checksum) {
        printk("ERROR: checksum mismatch gpu=%d cpu=%d\n", checksum, expected_checksum);
        mismatches++;
    }

    printk("VC4_TEST_RESULT name=saxpy_tmu status=%s mismatches=%d active_qpus=%d n=%d checksum=%d max_abs_diff=%f elapsed_usec=%d\n",
           (mismatches || launch_failures) ? "FAIL" : "PASS", mismatches,
           (int)SAXPY_TMU_ACTIVE_QPUS, (int)SAXPY_TMU_N, checksum, max_abs_diff, elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
