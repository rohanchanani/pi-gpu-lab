#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define WARP_REDUCE_SUM_EPSILON 0.0002f
#define CHECKSUM_SCALE 4096.0f
#define WARP_REDUCE_SUM_LANE_WIDTH 16u
#define WARP_REDUCE_SUM_MAX_QPUS 12u
#define WARP_REDUCE_SUM_GUARD_WORDS 32u
#define WARP_REDUCE_SUM_SENTINEL (-12345.0f)
#define WARP_REDUCE_SUM_MAX_N 511u
#define WARP_REDUCE_SUM_BUFFER_N (WARP_REDUCE_SUM_MAX_N + WARP_REDUCE_SUM_GUARD_WORDS)

static const uint32_t test_sizes[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 511u,
};

static float input_values[WARP_REDUCE_SUM_BUFFER_N];
static float out_values[WARP_REDUCE_SUM_BUFFER_N];
static float expected_values[WARP_REDUCE_SUM_BUFFER_N];

static float absf_local(float value) {
    return value < 0.0f ? -value : value;
}

static uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    if (a == 0u)
        return 0u;
    return (a + b - 1u) / b;
}

static float make_input_value(uint32_t i, uint32_t case_id) {
    int raw = (int)((i * 5u + case_id * 3u + 7u) % 17u) - 8;
    return (float)raw * 0.125f;
}

static void fill_buffers(uint32_t n, uint32_t case_id) {
    for (uint32_t i = 0; i < WARP_REDUCE_SUM_BUFFER_N; i++) {
        input_values[i] = 0.0f;
        out_values[i] = WARP_REDUCE_SUM_SENTINEL;
        expected_values[i] = WARP_REDUCE_SUM_SENTINEL;
    }

    for (uint32_t i = 0; i < n; i++)
        input_values[i] = make_input_value(i, case_id);
}

static void run_cpu_reference(uint32_t n) {
    uint32_t vectors = ceil_div_u32(n, WARP_REDUCE_SUM_LANE_WIDTH);

    for (uint32_t v = 0; v < vectors; v++) {
        uint32_t base = v * WARP_REDUCE_SUM_LANE_WIDTH;
        uint32_t remaining = n - base;
        uint32_t active = remaining < WARP_REDUCE_SUM_LANE_WIDTH ? remaining : WARP_REDUCE_SUM_LANE_WIDTH;
        float sum = 0.0f;

        for (uint32_t lane = 0; lane < active; lane++)
            sum += input_values[base + lane];
        for (uint32_t lane = 0; lane < active; lane++)
            expected_values[base + lane] = sum;
    }
}

static void verify_results(uint32_t n, int *mismatches, float *max_abs_diff) {
    *mismatches = 0;
    *max_abs_diff = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        float diff = out_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;
        if (abs_diff > WARP_REDUCE_SUM_EPSILON) {
            if (*mismatches < 8) {
                uint32_t vector = i / WARP_REDUCE_SUM_LANE_WIDTH;
                uint32_t lane = i % WARP_REDUCE_SUM_LANE_WIDTH;
                printk("ERROR: vector=%d lane=%d i=%d gpu=%f cpu=%f diff=%f\n",
                       (int)vector, (int)lane, (int)i, out_values[i],
                       expected_values[i], diff);
            }
            (*mismatches)++;
        }
    }
}

static int verify_sentinel_tail(uint32_t n) {
    int mismatches = 0;

    for (uint32_t i = n; i < n + WARP_REDUCE_SUM_GUARD_WORDS && i < WARP_REDUCE_SUM_BUFFER_N; i++) {
        if (out_values[i] != WARP_REDUCE_SUM_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: sentinel changed i=%d value=%f expected=%f\n",
                       (int)i, out_values[i], WARP_REDUCE_SUM_SENTINEL);
            mismatches++;
        }
    }

    return mismatches;
}

static int scaled_checksum(const float *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = WARP_REDUCE_SUM_BUFFER_N * sizeof(float);

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4Malloc(program, &input_dev, bytes) < 0 || vc4Malloc(program, &out_dev, bytes) < 0)
        panic("warp_reduce_sum device allocation failed");

    uint32_t active_qpus = WARP_REDUCE_SUM_MAX_QPUS;
    uint32_t lane_width = WARP_REDUCE_SUM_LANE_WIDTH;
    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(active_qpus * lane_width, 1u, 1u);
    const uint32_t case_count = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printk("WARP_REDUCE_SUM_RUNTIME_SETUP max_n=%d allocations=%d\n",
           WARP_REDUCE_SUM_MAX_N, 1);

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();

    for (uint32_t case_index = 0; case_index < case_count; case_index++) {
        uint32_t n = test_sizes[case_index];
        uint32_t vectors = ceil_div_u32(n, WARP_REDUCE_SUM_LANE_WIDTH);

        fill_buffers(n, case_index);
        run_cpu_reference(n);

        if (vc4MemcpyHtoD(program, input_dev, input_values, bytes) < 0 ||
            vc4MemcpyHtoD(program, out_dev, out_values, bytes) < 0 ||
            warp_reduce_sum_launch(program, grid, block, input_dev, out_dev, n) < 0 ||
            vc4MemcpyDtoH(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: warp_reduce_sum launch/copy failed case=%d n=%d\n",
                   (int)case_index, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(n, &mismatches, &max_abs_diff);
        int case_sentinel_mismatches = verify_sentinel_tail(n);
        int checksum = scaled_checksum(out_values, n);
        int expected_checksum = scaled_checksum(expected_values, n);
        if (checksum != expected_checksum) {
            printk("ERROR: checksum mismatch case=%d gpu=%d cpu=%d\n",
                   (int)case_index, checksum, expected_checksum);
            mismatches++;
        }

        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel_mismatches;
        checksum_accum += checksum;

        printk("WARP_REDUCE_SUM_CASE case=%d n=%d vectors=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
               (int)case_index, (int)n, (int)vectors, mismatches,
               case_sentinel_mismatches, checksum, max_abs_diff,
               (int)(case_index + 1u), 1);
    }

    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 &&
         sentinel_mismatches == 0 &&
         launch_failures == 0 &&
         max_abs_diff_overall <= WARP_REDUCE_SUM_EPSILON) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=warp_reduce_sum status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)case_count, total_mismatches, sentinel_mismatches,
           launch_failures, (int)active_qpus, (int)lane_width,
           WARP_REDUCE_SUM_MAX_N, checksum_accum, max_abs_diff_overall,
           1, (int)case_count, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
