#include "rpi.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SAXPY_BASIC_EPSILON 0.0001f
#define CHECKSUM_SCALE 1024.0f
#define EXPECTED_CHECKSUM 2834240
#define SAXPY_BASIC_ACTIVE_QPUS 12u
#define SAXPY_BASIC_LANE_WIDTH 16u
#define SAXPY_BASIC_N (SAXPY_BASIC_ACTIVE_QPUS * SAXPY_BASIC_LANE_WIDTH)

static float x_values[SAXPY_BASIC_N];
static float y_values[SAXPY_BASIC_N];
static float y_initial[SAXPY_BASIC_N];
static float expected_values[SAXPY_BASIC_N];

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
        if (abs_diff > SAXPY_BASIC_EPSILON) {
            if (*mismatch_count < 8)
                printk("ERROR: i=%d gpu=%f cpu=%f diff=%f\n", (int)i, y_values[i], expected_values[i], diff);
            (*mismatch_count)++;
        }
    }
}

void notmain(void) {
    const float alpha = 2.5f;
    const uint32_t activeQpus = SAXPY_BASIC_ACTIVE_QPUS;
    const uint32_t laneWidth = SAXPY_BASIC_LANE_WIDTH;
    const uint32_t n = SAXPY_BASIC_N;
    struct vc4_program *program = 0;

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    fill_inputs(n);
    run_cpu_reference(alpha, n);

    vc4_deviceptr_t x_dev = 0, y_dev = 0;
    uint32_t bytes = n * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0 ||
        vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
        vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0)
        panic("saxpy_basic device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

    printk("Running VC4 saxpy_basic M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launchFailures = 0;
    if (saxpy_basic_launch(program, grid, block, x_dev, y_dev, alpha, n) < 0 ||
        vc4_m2_copy_dtoh(program, y_values, y_dev, bytes) < 0)
        launchFailures++;
    int elapsed = timer_get_usec() - start;

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(n, &mismatches, &max_abs_diff);

    int checksum = scaled_checksum(y_values, n);
    int expected_checksum = scaled_checksum(expected_values, n);
    if (checksum != expected_checksum) {
        printk("ERROR: checksum mismatch gpu=%d cpu=%d\n", checksum, expected_checksum);
        mismatches++;
    }
    if (checksum != EXPECTED_CHECKSUM) {
        printk("ERROR: unexpected checksum got=%d expected=%d\n", checksum, EXPECTED_CHECKSUM);
        mismatches++;
    }

    const char *status = (mismatches == 0 && launchFailures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=saxpy_basic status=%s mismatches=%d active_qpus=%d n=%d checksum=%d max_abs_diff=%f elapsed_usec=%d\n",
           status, mismatches, (int)activeQpus, (int)n, checksum, max_abs_diff, elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
