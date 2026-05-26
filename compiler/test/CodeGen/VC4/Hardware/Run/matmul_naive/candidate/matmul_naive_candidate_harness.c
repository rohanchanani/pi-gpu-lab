#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MATMUL_NAIVE_MAX_M 12u
#define MATMUL_NAIVE_MAX_N 16u
#define MATMUL_NAIVE_MAX_K 4u
#define MATMUL_NAIVE_CASES 8u
#define MATMUL_NAIVE_A_COUNT (MATMUL_NAIVE_MAX_M * MATMUL_NAIVE_MAX_K)
#define MATMUL_NAIVE_B_COUNT (MATMUL_NAIVE_MAX_K * MATMUL_NAIVE_MAX_N)
#define MATMUL_NAIVE_C_CAPACITY (MATMUL_NAIVE_MAX_M * MATMUL_NAIVE_MAX_N)
#define MATMUL_NAIVE_SCRATCH_WORDS MATMUL_NAIVE_C_CAPACITY
#define MATMUL_NAIVE_GUARD_WORDS 16u
#define MATMUL_NAIVE_C_TOTAL (MATMUL_NAIVE_C_CAPACITY + MATMUL_NAIVE_SCRATCH_WORDS + MATMUL_NAIVE_GUARD_WORDS)
#define MATMUL_NAIVE_GUARD_BASE 0x6d6d0000u
#define CHECKSUM_SCALE 4096.0f

struct matmul_case {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

static const struct matmul_case cases[MATMUL_NAIVE_CASES] = {
    {12u, 16u, 4u},
    {7u, 13u, 3u},
    {5u, 5u, 1u},
    {3u, 16u, 0u},
    {1u, 1u, 4u},
    {11u, 9u, 2u},
    {2u, 15u, 3u},
    {9u, 2u, 4u},
};

static float a_values[MATMUL_NAIVE_A_COUNT];
static float b_values[MATMUL_NAIVE_B_COUNT];
static float c_values[MATMUL_NAIVE_C_TOTAL];
static float expected_values[MATMUL_NAIVE_C_CAPACITY];

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
    return MATMUL_NAIVE_GUARD_BASE ^ (case_index << 8) ^ word;
}

static float make_a_value(uint32_t case_index, uint32_t row, uint32_t kk) {
    int centered = (int)((row * 7u + kk * 3u + case_index * 5u) % 17u) - 8;
    return ((float)centered) * 0.125f;
}

static float make_b_value(uint32_t case_index, uint32_t kk, uint32_t col) {
    int centered = (int)((kk * 11u + col * 5u + case_index * 7u) % 19u) - 9;
    return ((float)centered) * 0.0625f;
}

static void fill_inputs(uint32_t case_index, const struct matmul_case *tc) {
    for (uint32_t i = 0; i < MATMUL_NAIVE_A_COUNT; i++)
        a_values[i] = 0.0f;
    for (uint32_t i = 0; i < MATMUL_NAIVE_B_COUNT; i++)
        b_values[i] = 0.0f;

    for (uint32_t row = 0; row < tc->m; row++)
        for (uint32_t kk = 0; kk < tc->k; kk++)
            a_values[row * tc->k + kk] = make_a_value(case_index, row, kk);

    for (uint32_t kk = 0; kk < tc->k; kk++)
        for (uint32_t col = 0; col < tc->n; col++)
            b_values[kk * tc->n + col] = make_b_value(case_index, kk, col);

    for (uint32_t i = 0; i < MATMUL_NAIVE_C_TOTAL; i++)
        c_values[i] = float_from_bits(guard_word(case_index, i));
    for (uint32_t i = 0; i < MATMUL_NAIVE_C_CAPACITY; i++)
        expected_values[i] = float_from_bits(guard_word(case_index, i));
}

static void run_cpu_reference(const struct matmul_case *tc) {
    for (uint32_t row = 0; row < tc->m; row++) {
        for (uint32_t col = 0; col < tc->n; col++) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < tc->k; kk++)
                acc += a_values[row * tc->k + kk] * b_values[kk * tc->n + col];
            expected_values[row * tc->n + col] = acc;
        }
    }
}

static int scaled_checksum(const float *values, uint32_t count) {
    int checksum = 0;
    for (uint32_t i = 0; i < count; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t case_index,
                           const struct matmul_case *tc,
                           int *mismatch_count,
                           int *guard_mismatches,
                           float *max_abs_diff) {
    uint32_t live_count = tc->m * tc->n;
    *mismatch_count = 0;
    *guard_mismatches = 0;
    *max_abs_diff = 0.0f;

    for (uint32_t i = 0; i < live_count; i++) {
        float diff = c_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;
        if (float_bits(c_values[i]) != float_bits(expected_values[i])) {
            if (*mismatch_count < 8) {
                uint32_t row = i / tc->n;
                uint32_t col = i % tc->n;
                printk("ERROR: case=%d row=%d col=%d gpu=%f cpu=%f gpu_bits=%x cpu_bits=%x diff=%f\n",
                       (int)case_index, (int)row, (int)col, c_values[i],
                       expected_values[i], float_bits(c_values[i]),
                       float_bits(expected_values[i]), diff);
            }
            (*mismatch_count)++;
        }
    }

    for (uint32_t i = live_count; i < MATMUL_NAIVE_C_CAPACITY; i++) {
        uint32_t actual = float_bits(c_values[i]);
        uint32_t expected = guard_word(case_index, i);
        if (actual != expected) {
            if (*guard_mismatches < 8)
                printk("ERROR: case=%d untouched_word=%d actual=0x%x expected=0x%x\n",
                       (int)case_index, (int)i, actual, expected);
            (*guard_mismatches)++;
        }
    }

    for (uint32_t i = MATMUL_NAIVE_C_CAPACITY + MATMUL_NAIVE_SCRATCH_WORDS;
         i < MATMUL_NAIVE_C_TOTAL; i++) {
        uint32_t actual = float_bits(c_values[i]);
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

    vc4_deviceptr_t a_dev = 0, b_dev = 0, c_dev = 0;
    uint32_t a_bytes = MATMUL_NAIVE_A_COUNT * sizeof(float);
    uint32_t b_bytes = MATMUL_NAIVE_B_COUNT * sizeof(float);
    uint32_t c_bytes = MATMUL_NAIVE_C_TOTAL * sizeof(float);
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &b_dev, b_bytes) < 0 ||
        vc4_m2_malloc(program, &c_dev, c_bytes) < 0)
        panic("matmul_naive device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(MATMUL_NAIVE_C_CAPACITY, 1, 1);

    printk("Running VC4 matmul_naive M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launch_failures = 0;
    int total_mismatches = 0;
    int total_guard_mismatches = 0;
    int checksum_accum = 0;
    int expected_checksum_accum = 0;
    uint32_t checked_elements = 0;
    float max_abs_diff_overall = 0.0f;

    for (uint32_t case_index = 0; case_index < MATMUL_NAIVE_CASES; case_index++) {
        const struct matmul_case *tc = &cases[case_index];
        uint32_t live_count = tc->m * tc->n;
        fill_inputs(case_index, tc);
        run_cpu_reference(tc);

        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, b_dev, b_values, b_bytes) < 0 ||
            vc4_m2_copy_htod(program, c_dev, c_values, c_bytes) < 0 ||
            matmul_naive_launch(program, grid, block, a_dev, b_dev, c_dev,
                                tc->m, tc->n, tc->k) < 0 ||
            vc4_m2_copy_dtoh(program, c_values, c_dev, c_bytes) < 0) {
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        int guard_mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(case_index, tc, &mismatches, &guard_mismatches, &max_abs_diff);
        int checksum = scaled_checksum(c_values, live_count);
        int expected_checksum = scaled_checksum(expected_values, live_count);
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
        checked_elements += live_count;
        printk("MATMUL_NAIVE_CASE case=%d m=%d n=%d k=%d elements=%d mismatches=%d guard_mismatches=%d checksum=%d expected_checksum=%d max_abs_diff=%f\n",
               (int)case_index, (int)tc->m, (int)tc->n, (int)tc->k,
               (int)live_count, mismatches, guard_mismatches, checksum,
               expected_checksum, max_abs_diff);
    }
    int elapsed = timer_get_usec() - start;

    uint32_t runtime_allocations = matmul_naive_runtime_allocations();
    uint32_t runtime_launches = matmul_naive_runtime_launches();
    uint32_t runtime_capacity = matmul_naive_runtime_capacity();
    uint32_t code_uploads = matmul_naive_runtime_code_uploads();
    uint32_t recorded_launch_failures = matmul_naive_runtime_launch_failures();

    const char *status = (total_mismatches == 0 &&
                          total_guard_mismatches == 0 &&
                          checksum_accum == expected_checksum_accum &&
                          launch_failures == 0 &&
                          recorded_launch_failures == 0 &&
                          runtime_allocations == 1u &&
                          runtime_launches == MATMUL_NAIVE_CASES &&
                          code_uploads == 1u) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=matmul_naive status=%s cases=%d checked_elements=%d total_mismatches=%d guard_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d max_m=%d max_n=%d max_k=%d scratch_words=%d checksum_accum=%d expected_checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_capacity=%d code_uploads=%d elapsed_usec=%d\n",
           status,
           (int)MATMUL_NAIVE_CASES,
           (int)checked_elements,
           total_mismatches,
           total_guard_mismatches,
           launch_failures,
           (int)recorded_launch_failures,
           (int)MATMUL_NAIVE_MAX_M,
           (int)MATMUL_NAIVE_MAX_N,
           (int)MATMUL_NAIVE_MAX_M,
           (int)MATMUL_NAIVE_MAX_N,
           (int)MATMUL_NAIVE_MAX_K,
           (int)MATMUL_NAIVE_SCRATCH_WORDS,
           checksum_accum,
           expected_checksum_accum,
           max_abs_diff_overall,
           (int)runtime_allocations,
           (int)runtime_launches,
           (int)runtime_capacity,
           (int)code_uploads,
           elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, c_dev);
    vc4_program_destroy(program);
}
