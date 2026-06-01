#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SAXPY_TAIL_VC4KERNEL_EPSILON 0.0001f
#define SAXPY_TAIL_VC4KERNEL_CHECKSUM_SCALE 1024.0f
#define SAXPY_TAIL_VC4KERNEL_MAX_N 1000u
#define SAXPY_TAIL_VC4KERNEL_GUARD 32u
#define SAXPY_TAIL_VC4KERNEL_ACTIVE_QPUS 12u
#define SAXPY_TAIL_VC4KERNEL_LANES 16u
#define SAXPY_TAIL_VC4KERNEL_ELEMENTS_PER_WAVE (SAXPY_TAIL_VC4KERNEL_ACTIVE_QPUS * SAXPY_TAIL_VC4KERNEL_LANES)
#define SAXPY_TAIL_VC4KERNEL_MAX_WAVES ((SAXPY_TAIL_VC4KERNEL_MAX_N + SAXPY_TAIL_VC4KERNEL_ELEMENTS_PER_WAVE - 1u) / SAXPY_TAIL_VC4KERNEL_ELEMENTS_PER_WAVE)
#define SAXPY_TAIL_VC4KERNEL_MAX_COVERAGE_N (SAXPY_TAIL_VC4KERNEL_MAX_WAVES * SAXPY_TAIL_VC4KERNEL_ELEMENTS_PER_WAVE)
#define SAXPY_TAIL_VC4KERNEL_BUFFER_N (SAXPY_TAIL_VC4KERNEL_MAX_COVERAGE_N + SAXPY_TAIL_VC4KERNEL_GUARD)
#define SAXPY_TAIL_VC4KERNEL_SENTINEL (-12345.0f)

static const uint32_t test_sizes[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 191u,
    192u, 193u, 255u, 256u, 257u, 767u, 768u, 769u, 1000u
};

static float x_values[SAXPY_TAIL_VC4KERNEL_BUFFER_N];
static float y_values[SAXPY_TAIL_VC4KERNEL_BUFFER_N];
static float y_initial[SAXPY_TAIL_VC4KERNEL_BUFFER_N];
static float expected_values[SAXPY_TAIL_VC4KERNEL_BUFFER_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + SAXPY_TAIL_VC4KERNEL_ELEMENTS_PER_WAVE - 1u) /
                     SAXPY_TAIL_VC4KERNEL_ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static void fill_inputs(uint32_t n) {
    for (uint32_t i = 0; i < SAXPY_TAIL_VC4KERNEL_BUFFER_N; i++) {
        x_values[i] = ((float)((i * 7u + 3u) % 101u) * 0.125f) - 4.0f;
        y_values[i] = ((float)((i * 5u + 11u) % 67u) * 0.25f) + 0.5f;
        y_initial[i] = y_values[i];
        expected_values[i] = y_values[i];
    }
    for (uint32_t i = n; i < SAXPY_TAIL_VC4KERNEL_BUFFER_N; i++) {
        y_values[i] = SAXPY_TAIL_VC4KERNEL_SENTINEL;
        y_initial[i] = SAXPY_TAIL_VC4KERNEL_SENTINEL;
        expected_values[i] = SAXPY_TAIL_VC4KERNEL_SENTINEL;
    }
}

static void run_cpu_reference(float alpha, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        expected_values[i] = alpha * x_values[i] + y_initial[i];
}

static int scaled_checksum(const float *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] * SAXPY_TAIL_VC4KERNEL_CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t n, int *mismatch_count, float *max_abs_diff) {
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float diff = y_values[i] - expected_values[i];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > SAXPY_TAIL_VC4KERNEL_EPSILON) {
            if (*mismatch_count < 8)
                printk("ERROR: saxpy_tail_vc4kernel n=%d i=%d gpu=%f cpu=%f diff=%f\n",
                       (int)n, (int)i, y_values[i], expected_values[i], diff);
            (*mismatch_count)++;
        }
    }
}

static int verify_sentinel_region(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < SAXPY_TAIL_VC4KERNEL_BUFFER_N; i++) {
        if (y_values[i] != SAXPY_TAIL_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: saxpy_tail_vc4kernel sentinel changed n=%d i=%d value=%f expected=%f\n",
                       (int)n, (int)i, y_values[i],
                       SAXPY_TAIL_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    const float alpha = 2.5f;
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = SAXPY_TAIL_VC4KERNEL_BUFFER_N * sizeof(float);
    vc4_deviceptr_t x_dev = 0;
    vc4_deviceptr_t y_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0)
        panic("saxpy_tail_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(SAXPY_TAIL_VC4KERNEL_ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 saxpy_tail_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(test_sizes) / sizeof(test_sizes[0]); case_id++) {
        uint32_t n = test_sizes[case_id];
        uint32_t waves = rounded_waves(n);
        uint32_t rounded_coverage = waves * SAXPY_TAIL_VC4KERNEL_ELEMENTS_PER_WAVE;
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_inputs(n);
        run_cpu_reference(alpha, n);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
            saxpy_tail_vc4kernel_launch(program, grid, block, x_dev, y_dev, alpha, n) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, bytes) < 0) {
            printk("ERROR: saxpy_tail_vc4kernel launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(n, &mismatches, &max_abs_diff);
        int case_sentinels = verify_sentinel_region(n);
        int checksum = scaled_checksum(y_values, n);
        int expected_checksum = scaled_checksum(expected_values, n);
        if (checksum != expected_checksum) {
            printk("ERROR: saxpy_tail_vc4kernel checksum mismatch n=%d gpu=%d cpu=%d\n",
                   (int)n, checksum, expected_checksum);
            mismatches++;
        }
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinels;
        checksum_accum += checksum;
        printk("SAXPY_TAIL_VC4KERNEL_CASE case=%d n=%d waves=%d coverage=%d buffer_n=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, (int)n, (int)waves, (int)rounded_coverage,
               SAXPY_TAIL_VC4KERNEL_BUFFER_N, mismatches, case_sentinels,
               checksum, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=saxpy_tail_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(test_sizes) / sizeof(test_sizes[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           SAXPY_TAIL_VC4KERNEL_ACTIVE_QPUS, SAXPY_TAIL_VC4KERNEL_LANES,
           SAXPY_TAIL_VC4KERNEL_MAX_N, SAXPY_TAIL_VC4KERNEL_MAX_COVERAGE_N,
           SAXPY_TAIL_VC4KERNEL_BUFFER_N, checksum_accum,
           max_abs_diff_overall, 2,
           (int)(sizeof(test_sizes) / sizeof(test_sizes[0])), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
