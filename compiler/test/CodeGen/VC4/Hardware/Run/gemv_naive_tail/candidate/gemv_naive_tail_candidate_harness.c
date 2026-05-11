#include "rpi.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GEMV_NAIVE_TAIL_CASES 9u
#define GEMV_NAIVE_TAIL_MAX_M 25u
#define GEMV_NAIVE_TAIL_MAX_N 33u
#define GEMV_NAIVE_TAIL_PADDED_M (((GEMV_NAIVE_TAIL_MAX_M + 15u) / 16u) * 16u)
#define GEMV_NAIVE_TAIL_MAX_A_WORDS (GEMV_NAIVE_TAIL_PADDED_M * GEMV_NAIVE_TAIL_MAX_N)
#define GEMV_NAIVE_TAIL_X_WORDS GEMV_NAIVE_TAIL_MAX_N
#define GEMV_NAIVE_TAIL_GUARD_WORDS 64u
#define GEMV_NAIVE_TAIL_SENTINEL (-9876.5f)
#define GEMV_ACTIVE_QPUS 12u
#define GEMV_LANE_WIDTH 16u

struct gemv_naive_tail_case { uint32_t m; uint32_t n; };

static const struct gemv_naive_tail_case gemv_cases[GEMV_NAIVE_TAIL_CASES] = {
    {0u, 7u}, {4u, 0u}, {1u, 1u}, {3u, 5u}, {5u, 16u}, {7u, 17u}, {12u, 31u}, {19u, 13u}, {25u, 33u},
};

static float a_values[GEMV_NAIVE_TAIL_MAX_A_WORDS];
static float x_values[GEMV_NAIVE_TAIL_X_WORDS];
static float y_values[GEMV_NAIVE_TAIL_MAX_M + GEMV_NAIVE_TAIL_GUARD_WORDS];
static float expected_values[GEMV_NAIVE_TAIL_MAX_M];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }

static float make_a_value(uint32_t row, uint32_t col, uint32_t case_id) {
    const int raw = (int)((row * 11u + col * 7u + case_id * 5u + 3u) % 17u) - 8;
    return (float)raw * 0.125f;
}

static float make_x_value(uint32_t col, uint32_t case_id) {
    const int raw = (int)((col * 13u + case_id * 3u + 1u) % 19u) - 9;
    return (float)raw * 0.25f;
}

static void fill_case(uint32_t case_id, uint32_t m, uint32_t n) {
    for (uint32_t i = 0; i < GEMV_NAIVE_TAIL_MAX_A_WORDS; i++)
        a_values[i] = 0.0f;
    for (uint32_t i = 0; i < GEMV_NAIVE_TAIL_X_WORDS; i++)
        x_values[i] = make_x_value(i, case_id);
    for (uint32_t i = 0; i < GEMV_NAIVE_TAIL_MAX_M; i++)
        expected_values[i] = 0.0f;
    for (uint32_t i = 0; i < GEMV_NAIVE_TAIL_MAX_M + GEMV_NAIVE_TAIL_GUARD_WORDS; i++)
        y_values[i] = GEMV_NAIVE_TAIL_SENTINEL;

    for (uint32_t row = 0; row < m; row++)
        for (uint32_t col = 0; col < n; col++)
            a_values[row * n + col] = make_a_value(row, col, case_id);

    for (uint32_t row = 0; row < m; row++) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < n; col++)
            sum += a_values[row * n + col] * x_values[col];
        expected_values[row] = sum;
    }
}

void notmain(void) {
    struct vc4_program *program = 0;
    uint32_t total_mismatches = 0u;
    uint32_t total_sentinel_mismatches = 0u;
    uint32_t launch_failures = 0u;
    float global_max_abs_diff = 0.0f;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t a_dev = 0, x_dev = 0, y_dev = 0;
    uint32_t a_bytes = GEMV_NAIVE_TAIL_MAX_A_WORDS * sizeof(float);
    uint32_t x_bytes = GEMV_NAIVE_TAIL_X_WORDS * sizeof(float);
    uint32_t y_bytes = (GEMV_NAIVE_TAIL_MAX_M + GEMV_NAIVE_TAIL_GUARD_WORDS) * sizeof(float);
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, y_bytes) < 0)
        panic("gemv_naive_tail device allocation failed");

    printk("GEMV_NAIVE_TAIL_RUNTIME_SETUP max_m=%d max_n=%d allocations=%d\n", GEMV_NAIVE_TAIL_MAX_M, GEMV_NAIVE_TAIL_MAX_N, 1);

    vc4_dim3 block = vc4_m2_dim3(GEMV_LANE_WIDTH, 1, 1);

    for (uint32_t case_id = 0; case_id < GEMV_NAIVE_TAIL_CASES; case_id++) {
        const uint32_t m = gemv_cases[case_id].m;
        const uint32_t n = gemv_cases[case_id].n;
        uint32_t mismatches = 0u;
        uint32_t sentinel_mismatches = 0u;
        float max_abs_diff = 0.0f;
        int checksum = 0;

        fill_case(case_id, m, n);
        vc4_dim3 grid = vc4_m2_dim3(m == 0u ? 0u : 1u, 1, 1);

        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, y_bytes) < 0 ||
            gemv_naive_tail_launch(program, grid, block, a_dev, x_dev, y_dev, m, n) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, y_bytes) < 0) {
            launch_failures++;
            printk("GEMV_NAIVE_TAIL_CASE case=%d m=%d n=%d launch=FAIL launches=%d allocations=%d\n", (int)case_id, (int)m, (int)n, (int)(case_id + 1), 1);
            continue;
        }

        for (uint32_t row = 0; row < m; row++) {
            const float diff = y_values[row] - expected_values[row];
            const float adiff = abs_f32(diff);
            checksum += (int)(y_values[row] * 1024.0f);
            if (adiff > max_abs_diff)
                max_abs_diff = adiff;
            if (adiff > 0.001f) {
                if (mismatches < 8u)
                    printk("ERROR: case=%d row=%d gpu=%f cpu=%f diff=%f\n", (int)case_id, (int)row, y_values[row], expected_values[row], diff);
                mismatches++;
            }
        }

        for (uint32_t i = m; i < GEMV_NAIVE_TAIL_MAX_M + GEMV_NAIVE_TAIL_GUARD_WORDS; i++)
            if (y_values[i] != GEMV_NAIVE_TAIL_SENTINEL) {
                if (sentinel_mismatches < 8u)
                    printk("ERROR: sentinel case=%d row=%d value=%f expected=%f\n", (int)case_id, (int)i, y_values[i], GEMV_NAIVE_TAIL_SENTINEL);
                sentinel_mismatches++;
            }

        if (max_abs_diff > global_max_abs_diff)
            global_max_abs_diff = max_abs_diff;
        total_mismatches += mismatches;
        total_sentinel_mismatches += sentinel_mismatches;
        checksum_accum += checksum;

        printk("GEMV_NAIVE_TAIL_CASE case=%d m=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
               (int)case_id, (int)m, (int)n, (int)mismatches, (int)sentinel_mismatches, checksum, max_abs_diff, (int)(case_id + 1), 1);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && total_sentinel_mismatches == 0u && launch_failures == 0u && global_max_abs_diff <= 0.001f) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=gemv_naive_tail status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_m=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, GEMV_NAIVE_TAIL_CASES, (int)total_mismatches, (int)total_sentinel_mismatches, (int)launch_failures, GEMV_ACTIVE_QPUS, GEMV_LANE_WIDTH, GEMV_NAIVE_TAIL_MAX_M, GEMV_NAIVE_TAIL_MAX_N, checksum_accum, global_max_abs_diff, 1, GEMV_NAIVE_TAIL_CASES, elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
