#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MAX_N 1000u
#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_WAVES ((MAX_N + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE)
#define MAX_COVERAGE_N (MAX_WAVES * ELEMENTS_PER_WAVE)
#define BUFFER_N (MAX_COVERAGE_N + 2u * GUARD)
#define SENTINEL_I32 ((int32_t)0x5a17c0de)

static const uint32_t n_cases[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u,
    63u, 64u, 65u, 127u, 128u, 129u, 193u, 1000u
};

static const int32_t bias_cases[] = {0, 11, -37};
static const int32_t threshold_cases[] = {-1000, 0, 4000};

static int32_t x_values[BUFFER_N];
static int32_t y_values[BUFFER_N];
static int32_t alt_values[BUFFER_N];
static int32_t out_values[BUFFER_N];
static int32_t expected_values[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static int32_t x_value(uint32_t i) {
    return (int32_t)((i * 7919u + 17u) % 100000u) - 50000;
}

static int32_t y_value(uint32_t i) {
    return (int32_t)((i * 3571u + 29u) % 80000u) - 40000;
}

static int32_t alt_value(uint32_t i) {
    return (int32_t)((i * 1237u + 101u) % 60000u) - 30000;
}

static void fill_buffers(uint32_t n, int32_t bias, int32_t threshold) {
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active = i - GUARD;
        x_values[i] = x_value(active);
        y_values[i] = y_value(active);
        alt_values[i] = alt_value(active);
        out_values[i] = SENTINEL_I32;
        expected_values[i] = SENTINEL_I32;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        int32_t tmp = x_values[index] + y_values[index] - bias;
        expected_values[index] = tmp > threshold ? tmp : alt_values[index];
    }
}

static int verify_results(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        int32_t got = out_values[index];
        int32_t expected = expected_values[index];
        if (got != expected) {
            if (mismatches < 8)
                printk("ERROR: triton_i32_add_select n=%d i=%d gpu=%d cpu=%d x=%d y=%d alt=%d\n",
                       (int)n, (int)i, (int)got, (int)expected,
                       (int)x_values[index], (int)y_values[index],
                       (int)alt_values[index]);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + n)
            continue;
        if (out_values[i] != SENTINEL_I32) {
            if (mismatches < 8)
                printk("ERROR: triton_i32_add_select sentinel n=%d i=%d got=%d expected=%d\n",
                       (int)n, (int)i, (int)out_values[i], (int)SENTINEL_I32);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= (uint32_t)out_values[index] + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t x_dev = 0, y_dev = 0, alt_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &alt_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("triton_i32_add_select allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t bias_id = 0; bias_id < sizeof(bias_cases) / sizeof(bias_cases[0]); bias_id++) {
        for (uint32_t n_id = 0; n_id < sizeof(n_cases) / sizeof(n_cases[0]); n_id++) {
            int32_t bias = bias_cases[bias_id];
            int32_t threshold = threshold_cases[bias_id];
            uint32_t n = n_cases[n_id];
            uint32_t waves = rounded_waves(n);
            uint32_t coverage = waves * ELEMENTS_PER_WAVE;
            vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
            fill_buffers(n, bias, threshold);

            vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(int32_t);
            vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(int32_t);
            vc4_deviceptr_t alt_active = alt_dev + GUARD * sizeof(int32_t);
            vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);
            if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, alt_dev, alt_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
                i32_add_select_b16_kernel_launch(program, grid, block, x_active,
                                                 y_active, alt_active, out_active,
                                                 bias, threshold, n) < 0 ||
                vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
                printk("ERROR: triton_i32_add_select launch/copy failed bias_id=%d n=%d\n",
                       (int)bias_id, (int)n);
                launch_failures++;
                continue;
            }

            int mismatches = verify_results(n);
            int sentinels = verify_sentinels(n);
            uint32_t case_hash = hash_output(n);
            output_hash ^= case_hash + 0x9e3779b9u + (bias_id << 12) + (n_id << 4);
            output_hash = rotl32_local(output_hash, 7u);
            total_mismatches += mismatches;
            sentinel_mismatches += sentinels;
            elements_checked += (int)n;
            printk("TRITON_I32_ADD_SELECT_TAIL_CASE bias_id=%d n=%d waves=%d coverage=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
                   (int)bias_id, (int)n, (int)waves, (int)coverage,
                   mismatches, sentinels, case_hash);
        }
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=triton_i32_add_select_tail_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u saw_ttir_import=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)((sizeof(bias_cases) / sizeof(bias_cases[0])) *
                         (sizeof(n_cases) / sizeof(n_cases[0]))),
           elements_checked, total_mismatches, sentinel_mismatches, launch_failures,
           ACTIVE_QPUS, LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N, output_hash,
           VC4_CASE_SAW_TTIR_IMPORT, 4,
           (int)((sizeof(bias_cases) / sizeof(bias_cases[0])) *
                 (sizeof(n_cases) / sizeof(n_cases[0]))), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, alt_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
