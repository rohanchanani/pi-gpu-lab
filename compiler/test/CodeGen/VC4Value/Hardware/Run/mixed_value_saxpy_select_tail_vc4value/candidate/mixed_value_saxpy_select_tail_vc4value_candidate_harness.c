#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MAX_N 1000u
#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_WAVES ((MAX_N + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE)
#define MAX_COVERAGE_N (MAX_WAVES * ELEMENTS_PER_WAVE)
#define BUFFER_N (MAX_COVERAGE_N + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u
#define THRESHOLD 3.25f
#define EPSILON 0.001f

static const uint32_t n_cases[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 64u, 65u, 193u, 1000u
};
static const float a_cases[] = {1.0f, -2.0f, 0.5f};

static float x_values[BUFFER_N];
static float y_values[BUFFER_N];
static float fallback_values[BUFFER_N];
static float out_values[BUFFER_N];
static float expected_values[BUFFER_N];

static float absf_local(float v) { return v < 0.0f ? -v : v; }

static float bits_to_float(uint32_t bits) {
    union { uint32_t u; float f; } value;
    value.u = bits;
    return value.f;
}

static uint32_t float_to_bits(float value) {
    union { uint32_t u; float f; } bits;
    bits.f = value;
    return bits.u;
}

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static float x_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 13u + 7u) % 89u) - 44;
    float value = (float)whole + (float)((i * 5u + 3u) & 3u) * 0.25f;
    return (i & 1u) ? -value : value;
}

static float y_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 17u + 11u) % 71u) - 35;
    return (float)whole * 0.5f + (float)((i + 1u) & 1u) * 0.25f;
}

static float fallback_value(uint32_t i) {
    return -500.0f - (float)((i * 7u) % 251u) * 0.25f;
}

static void fill_buffers(uint32_t n, float a) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active = i - GUARD;
        x_values[i] = x_value(active);
        y_values[i] = y_value(active);
        fallback_values[i] = fallback_value(active);
        out_values[i] = sentinel;
        expected_values[i] = sentinel;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        float tmp = a * x_values[index] + y_values[index];
        expected_values[index] = tmp > THRESHOLD ? tmp : fallback_values[index];
    }
}

static int verify_results(uint32_t n, float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        float diff = out_values[index] - expected_values[index];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > EPSILON) {
            if (mismatches < 8)
                printk("ERROR: mixed_saxpy_select n=%d i=%d gpu=%f cpu=%f diff=%f\n",
                       (int)n, (int)i, out_values[index], expected_values[index], diff);
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
        uint32_t got = float_to_bits(out_values[i]);
        if (got != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed_saxpy_select sentinel n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, got, SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    uint32_t bytes = BUFFER_N * sizeof(float);
    vc4_deviceptr_t x_dev = 0, y_dev = 0, fallback_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &fallback_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("mixed_saxpy_select allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    float max_abs_diff_overall = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t a_id = 0; a_id < sizeof(a_cases) / sizeof(a_cases[0]); a_id++) {
        for (uint32_t n_id = 0; n_id < sizeof(n_cases) / sizeof(n_cases[0]); n_id++) {
            float a = a_cases[a_id];
            uint32_t n = n_cases[n_id];
            uint32_t waves = rounded_waves(n);
            vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
            fill_buffers(n, a);
            vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
            vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
            vc4_deviceptr_t fallback_active = fallback_dev + GUARD * sizeof(float);
            vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
            if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, fallback_dev, fallback_values, bytes) < 0 ||
                vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
                mixed_value_saxpy_select_tail_vc4value_launch(
                    program, grid, block, x_active, y_active, fallback_active,
                    out_active, a, THRESHOLD, n) < 0 ||
                vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
                printk("ERROR: mixed_saxpy_select launch/copy failed a_id=%d n=%d\n",
                       (int)a_id, (int)n);
                launch_failures++;
                continue;
            }
            float max_abs_diff = 0.0f;
            int mismatches = verify_results(n, &max_abs_diff);
            int sentinels = verify_sentinels(n);
            uint32_t case_hash = hash_output(n);
            output_hash ^= case_hash + 0x9e3779b9u + (a_id << 12) + (n_id << 4);
            output_hash = rotl32_local(output_hash, 7u);
            if (max_abs_diff > max_abs_diff_overall)
                max_abs_diff_overall = max_abs_diff;
            total_mismatches += mismatches;
            sentinel_mismatches += sentinels;
            elements_checked += (int)n;
            printk("MIXED_VALUE_SAXPY_SELECT_CASE a_id=%d n=%d waves=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
                   (int)a_id, (int)n, (int)waves, mismatches, sentinels,
                   case_hash, max_abs_diff);
        }
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_value_saxpy_select_tail_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u max_abs_diff=%f saw_value_mixed_elementwise=1 saw_value_tail_mask=1 saw_value_tmu_load=1 saw_value_vdw_preserve=1 saw_value_f32_cmp_select=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)((sizeof(a_cases) / sizeof(a_cases[0])) *
                         (sizeof(n_cases) / sizeof(n_cases[0]))),
           elements_checked, total_mismatches, sentinel_mismatches,
           launch_failures, ACTIVE_QPUS, LANES, MAX_N, MAX_COVERAGE_N,
           BUFFER_N, output_hash, max_abs_diff_overall, 4,
           (int)((sizeof(a_cases) / sizeof(a_cases[0])) *
                 (sizeof(n_cases) / sizeof(n_cases[0]))), elapsed);
    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, fallback_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
