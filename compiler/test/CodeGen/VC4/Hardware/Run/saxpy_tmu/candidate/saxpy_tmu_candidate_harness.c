#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define CHECKSUM_SCALE 1024.0f
#define SAXPY_TMU_ACTIVE_QPUS 12u
#define SAXPY_TMU_LANE_WIDTH 16u
#define SAXPY_TMU_N (SAXPY_TMU_ACTIVE_QPUS * SAXPY_TMU_LANE_WIDTH)
#define SAXPY_TMU_GUARD_WORDS 16u
#define SAXPY_TMU_TOTAL_WORDS (SAXPY_TMU_N + SAXPY_TMU_GUARD_WORDS)
#define SAXPY_TMU_GUARD_BASE 0x5a7a0000u

static float x_values[SAXPY_TMU_TOTAL_WORDS];
static float y_values[SAXPY_TMU_TOTAL_WORDS];
static float y_initial[SAXPY_TMU_N];
static float expected_values[SAXPY_TMU_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_from_bits(uint32_t bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t guard_word(uint32_t case_index, uint32_t word) {
    return SAXPY_TMU_GUARD_BASE ^ (case_index << 8) ^ word;
}

static void fill_inputs(uint32_t case_index, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        x_values[i] = ((float)((i + case_index * 11u) % 97u) * 0.125f) - 3.0f;
        y_values[i] = ((float)((i * 3u + case_index * 7u) % 53u) * 0.25f) + 1.0f;
        y_initial[i] = y_values[i];
    }
    for (uint32_t i = 0; i < SAXPY_TMU_GUARD_WORDS; i++) {
        x_values[n + i] = float_from_bits(guard_word(case_index, i));
        y_values[n + i] = float_from_bits(guard_word(case_index, i));
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

static void verify_results(uint32_t case_index,
                           uint32_t n,
                           int *mismatch_count,
                           int *guard_mismatches,
                           float *max_abs_diff) {
    *mismatch_count = 0;
    *guard_mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float diff = y_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;
        if (float_bits(y_values[i]) != float_bits(expected_values[i])) {
            if (*mismatch_count < 8)
                printk("ERROR: case=%d i=%d gpu=%f cpu=%f gpu_bits=%x cpu_bits=%x diff=%f\n",
                       (int)case_index, (int)i, y_values[i], expected_values[i],
                       float_bits(y_values[i]), float_bits(expected_values[i]), diff);
            (*mismatch_count)++;
        }
    }
    for (uint32_t i = 0; i < SAXPY_TMU_GUARD_WORDS; i++) {
        uint32_t actual = float_bits(y_values[n + i]);
        uint32_t expected = guard_word(case_index, i);
        if (actual != expected) {
            if (*guard_mismatches < 8)
                printk("ERROR: case=%d guard=%d actual=0x%x expected=0x%x\n",
                       (int)case_index, (int)i, actual, expected);
            (*guard_mismatches)++;
        }
    }
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    const float alphas[] = {2.5f, -0.75f, 0.0f, 1.25f};
    const uint32_t case_count = sizeof(alphas) / sizeof(alphas[0]);

    vc4_deviceptr_t x_dev = 0, y_dev = 0;
    uint32_t bytes = SAXPY_TMU_TOTAL_WORDS * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0)
        panic("saxpy_tmu device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(SAXPY_TMU_N, 1, 1);

    printk("Running VC4 saxpy_tmu M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launch_failures = 0;
    int total_mismatches = 0;
    int total_guard_mismatches = 0;
    int checksum_accum = 0;
    int expected_checksum_accum = 0;
    float max_abs_diff_overall = 0.0f;

    for (uint32_t case_index = 0; case_index < case_count; case_index++) {
        float alpha = alphas[case_index];
        fill_inputs(case_index, SAXPY_TMU_N);
        run_cpu_reference(alpha, SAXPY_TMU_N);

        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
            saxpy_tmu_launch(program, grid, block, x_dev, y_dev, alpha, SAXPY_TMU_N) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, bytes) < 0) {
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        int guard_mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(case_index, SAXPY_TMU_N, &mismatches, &guard_mismatches, &max_abs_diff);
        int checksum = scaled_checksum(y_values, SAXPY_TMU_N);
        int expected_checksum = scaled_checksum(expected_values, SAXPY_TMU_N);
        if (checksum != expected_checksum) {
            printk("ERROR: case=%d checksum mismatch gpu=%d cpu=%d\n",
                   (int)case_index, checksum, expected_checksum);
            mismatches++;
        }
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        total_guard_mismatches += guard_mismatches;
        checksum_accum += checksum;
        expected_checksum_accum += expected_checksum;
        printk("SAXPY_TMU_CASE case=%d alpha=%f mismatches=%d guard_mismatches=%d checksum=%d expected_checksum=%d max_abs_diff=%f\n",
               (int)case_index, alpha, mismatches, guard_mismatches, checksum,
               expected_checksum, max_abs_diff);
    }
    int elapsed = timer_get_usec() - start;

    uint32_t runtime_allocations = saxpy_tmu_runtime_allocations();
    uint32_t runtime_launches = saxpy_tmu_runtime_launches();
    uint32_t runtime_capacity = saxpy_tmu_runtime_capacity();
    uint32_t code_uploads = saxpy_tmu_runtime_code_uploads();
    uint32_t recorded_launch_failures = saxpy_tmu_runtime_launch_failures();

    const char *status = (total_mismatches == 0 &&
                          total_guard_mismatches == 0 &&
                          checksum_accum == expected_checksum_accum &&
                          launch_failures == 0 &&
                          recorded_launch_failures == 0 &&
                          runtime_allocations == 1u &&
                          runtime_launches == case_count &&
                          code_uploads == 1u) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=saxpy_tmu status=%s cases=%d checked_elements=%d total_mismatches=%d guard_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d n=%d checksum_accum=%d expected_checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_capacity=%d code_uploads=%d elapsed_usec=%d\n",
           status,
           (int)case_count,
           (int)(case_count * SAXPY_TMU_N),
           total_mismatches,
           total_guard_mismatches,
           launch_failures,
           (int)recorded_launch_failures,
           (int)SAXPY_TMU_ACTIVE_QPUS,
           (int)SAXPY_TMU_LANE_WIDTH,
           (int)SAXPY_TMU_N,
           checksum_accum,
           expected_checksum_accum,
           max_abs_diff_overall,
           (int)runtime_allocations,
           (int)runtime_launches,
           (int)runtime_capacity,
           (int)code_uploads,
           elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
