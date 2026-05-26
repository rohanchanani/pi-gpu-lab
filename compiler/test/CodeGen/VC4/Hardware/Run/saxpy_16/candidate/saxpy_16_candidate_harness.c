#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SAXPY_16_CHUNKS_PER_QPU 4u
#define SAXPY_16_EPSILON 0.0001f
#define CHECKSUM_SCALE 1024.0f
#define EXPECTED_CHECKSUM 11592704
#define SAXPY_16_ACTIVE_QPUS 12u
#define SAXPY_16_LANE_WIDTH 16u
#define SAXPY_16_MAX_N (SAXPY_16_ACTIVE_QPUS * SAXPY_16_LANE_WIDTH * SAXPY_16_CHUNKS_PER_QPU)
#define SAXPY_16_GUARD_WORDS 16u
#define SAXPY_16_SENTINEL_BASE 12345.0f

static float x_values[SAXPY_16_MAX_N];
static float y_values[SAXPY_16_MAX_N + SAXPY_16_GUARD_WORDS];
static float y_initial[SAXPY_16_MAX_N];
static float expected_values[SAXPY_16_MAX_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static void fill_inputs(uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        x_values[i] = ((float)(i % 97u) * 0.125f) - 3.0f;
        y_values[i] = ((float)(i % 53u) * 0.25f) + 1.0f;
        y_initial[i] = y_values[i];
    }
    for (uint32_t i = n; i < n + SAXPY_16_GUARD_WORDS; i++)
        y_values[i] = SAXPY_16_SENTINEL_BASE + (float)i;
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
        if (abs_diff > SAXPY_16_EPSILON) {
            if (*mismatch_count < 8)
                printk("ERROR: i=%d gpu=%f cpu=%f diff=%f\n", (int)i, y_values[i], expected_values[i], diff);
            (*mismatch_count)++;
        }
    }
}

static int verify_guard(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < n + SAXPY_16_GUARD_WORDS; i++) {
        float expected = SAXPY_16_SENTINEL_BASE + (float)i;
        float actual = y_values[i];
        if (actual != expected) {
            if (mismatches < 8)
                printk("ERROR: guard i=%d gpu=%f expected=%f\n", (int)i, actual, expected);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    const float alpha = 2.5f;
    const uint32_t activeQpus = SAXPY_16_ACTIVE_QPUS;
    const uint32_t laneWidth = SAXPY_16_LANE_WIDTH;
    const uint32_t n = SAXPY_16_MAX_N;
    struct vc4_program *program = 0;

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    fill_inputs(n);
    run_cpu_reference(alpha, n);

    vc4_deviceptr_t x_dev = 0, y_dev = 0;
    uint32_t x_bytes = n * sizeof(float);
    uint32_t y_bytes = (n + SAXPY_16_GUARD_WORDS) * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, y_bytes) < 0 ||
        vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
        vc4_m2_copy_htod(program, y_dev, y_values, y_bytes) < 0)
        panic("saxpy_16 device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

    printk("Running VC4 saxpy_16 M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launchFailures = 0;
    if (saxpy_16_launch(program, grid, block, x_dev, y_dev, alpha, n) < 0 ||
        vc4_m2_copy_dtoh(program, y_values, y_dev, y_bytes) < 0)
        launchFailures++;
    int elapsed = timer_get_usec() - start;

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(n, &mismatches, &max_abs_diff);
    int sentinel_mismatches = verify_guard(n);

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

    uint32_t runtime_allocations = saxpy_16_runtime_allocations();
    uint32_t runtime_launches = saxpy_16_runtime_launches();

    const char *status =
        (mismatches == 0 && sentinel_mismatches == 0 && launchFailures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=saxpy_16 status=%s checked_elements=%d mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d n=%d chunks_per_qpu=%d checksum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)n, mismatches, sentinel_mismatches, launchFailures,
           (int)activeQpus, (int)laneWidth, (int)n, (int)SAXPY_16_CHUNKS_PER_QPU,
           checksum, max_abs_diff, (int)runtime_allocations, (int)runtime_launches, elapsed);
    if (mismatches || sentinel_mismatches || launchFailures)
        panic("saxpy_16 verification failed: mismatches=%d sentinel_mismatches=%d launch_failures=%d",
              mismatches, sentinel_mismatches, launchFailures);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
