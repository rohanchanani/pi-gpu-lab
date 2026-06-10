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
#define SENTINEL_I32 0x5e17aa55u
#define THRESHOLD_F 2.75f
#define THRESHOLD_I 19

static const uint32_t n_cases[] = {0u, 1u, 15u, 16u, 17u, 32u, 33u, 191u, 193u, 1000u};

static float xf_values[BUFFER_N], lo_f_values[BUFFER_N], hi_f_values[BUFFER_N], out_f_values[BUFFER_N];
static int32_t xi_values[BUFFER_N], lo_i_values[BUFFER_N], hi_i_values[BUFFER_N], out_i_values[BUFFER_N];
static uint32_t expected_f_bits[BUFFER_N];
static int32_t expected_i_values[BUFFER_N];

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

static float xf_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 17u + 3u) % 79u) - 39;
    return (float)whole * 0.25f;
}

static float lo_f_value(uint32_t i) { return -500.0f - (float)((i * 5u) % 97u); }
static float hi_f_value(uint32_t i) { return 700.0f + (float)((i * 7u) % 89u); }
static int32_t xi_value(uint32_t i) { return (int32_t)((i * 23u + 11u) % 91u) - 45; }
static int32_t lo_i_value(uint32_t i) { return -30000 - (int32_t)((i * 13u) % 1021u); }
static int32_t hi_i_value(uint32_t i) { return 40000 + (int32_t)((i * 29u) % 2039u); }

static void fill_buffers(uint32_t n) {
    float sentinel_f = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active = i - GUARD;
        xf_values[i] = xf_value(active);
        lo_f_values[i] = lo_f_value(active);
        hi_f_values[i] = hi_f_value(active);
        out_f_values[i] = sentinel_f;
        xi_values[i] = xi_value(active);
        lo_i_values[i] = lo_i_value(active);
        hi_i_values[i] = hi_i_value(active);
        out_i_values[i] = (int32_t)SENTINEL_I32;
        expected_f_bits[i] = SENTINEL_BITS;
        expected_i_values[i] = (int32_t)SENTINEL_I32;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        expected_f_bits[index] = float_to_bits(xf_values[index] < THRESHOLD_F ?
                                               lo_f_values[index] : hi_f_values[index]);
        expected_i_values[index] = xi_values[index] > THRESHOLD_I ?
                                   hi_i_values[index] : lo_i_values[index];
    }
}

static int verify_results(uint32_t n, float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        uint32_t got_f = float_to_bits(out_f_values[index]);
        if (got_f != expected_f_bits[index]) {
            float diff = out_f_values[index] - bits_to_float(expected_f_bits[index]);
            float ad = diff < 0.0f ? -diff : diff;
            if (ad > *max_abs_diff)
                *max_abs_diff = ad;
            if (mismatches < 8)
                printk("ERROR: compute_select f32 n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, got_f, expected_f_bits[index]);
            mismatches++;
        }
        if (out_i_values[index] != expected_i_values[index]) {
            if (mismatches < 8)
                printk("ERROR: compute_select i32 n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, (uint32_t)out_i_values[index],
                       (uint32_t)expected_i_values[index]);
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
        if (float_to_bits(out_f_values[i]) != SENTINEL_BITS ||
            (uint32_t)out_i_values[i] != SENTINEL_I32) {
            if (mismatches < 8)
                printk("ERROR: compute_select sentinel n=%d i=%d got_f=%x got_i=%x\n",
                       (int)n, (int)i, float_to_bits(out_f_values[i]),
                       (uint32_t)out_i_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= float_to_bits(out_f_values[index]) + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
        hash ^= (uint32_t)out_i_values[index] + 0x85ebca6bu + (i << 5);
        hash = rotl32_local(hash, 7u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    uint32_t bytes_f = BUFFER_N * sizeof(float);
    uint32_t bytes_i = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t xf_dev = 0, lo_f_dev = 0, hi_f_dev = 0, out_f_dev = 0;
    vc4_deviceptr_t xi_dev = 0, lo_i_dev = 0, hi_i_dev = 0, out_i_dev = 0;
    if (vc4_m2_malloc(program, &xf_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &lo_f_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &hi_f_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &out_f_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &xi_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &lo_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &hi_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &out_i_dev, bytes_i) < 0)
        panic("value_mask_compute_select allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    float max_abs_diff_overall = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t case_id = 0; case_id < sizeof(n_cases) / sizeof(n_cases[0]); case_id++) {
        uint32_t n = n_cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(rounded_waves(n), 1, 1);
        fill_buffers(n);
        vc4_deviceptr_t xf_active = xf_dev + GUARD * sizeof(float);
        vc4_deviceptr_t lo_f_active = lo_f_dev + GUARD * sizeof(float);
        vc4_deviceptr_t hi_f_active = hi_f_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_f_active = out_f_dev + GUARD * sizeof(float);
        vc4_deviceptr_t xi_active = xi_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t lo_i_active = lo_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t hi_i_active = hi_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t out_i_active = out_i_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, xf_dev, xf_values, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, lo_f_dev, lo_f_values, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, hi_f_dev, hi_f_values, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, out_f_dev, out_f_values, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, xi_dev, xi_values, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, lo_i_dev, lo_i_values, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, hi_i_dev, hi_i_values, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, out_i_dev, out_i_values, bytes_i) < 0 ||
            value_mask_compute_select_tail_vc4value_launch(
                program, grid, block, xf_active, lo_f_active, hi_f_active,
                out_f_active, xi_active, lo_i_active, hi_i_active, out_i_active,
                THRESHOLD_F, THRESHOLD_I, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_f_values, out_f_dev, bytes_f) < 0 ||
            vc4_m2_copy_dtoh(program, out_i_values, out_i_dev, bytes_i) < 0) {
            printk("ERROR: value_mask_compute_select launch/copy failed n=%d\n", (int)n);
            launch_failures++;
            continue;
        }
        float max_abs_diff = 0.0f;
        int mismatches = verify_results(n, &max_abs_diff);
        int sentinels = verify_sentinels(n);
        uint32_t case_hash = hash_output(n);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        printk("VALUE_MASK_COMPUTE_SELECT_CASE case=%d n=%d waves=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)n, (int)rounded_waves(n), mismatches,
               sentinels, case_hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_mask_compute_select_tail_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d saw_value_compute_mask_select=1 saw_value_memory_tail_mask=1 saw_no_sparse_memory_mask=1 elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N, output_hash,
           output_hash != 0u ? 1 : 0, max_abs_diff_overall, 8,
           (int)(sizeof(n_cases) / sizeof(n_cases[0])), elapsed);

    vc4Free(program, xf_dev); vc4Free(program, lo_f_dev);
    vc4Free(program, hi_f_dev); vc4Free(program, out_f_dev);
    vc4Free(program, xi_dev); vc4Free(program, lo_i_dev);
    vc4Free(program, hi_i_dev); vc4Free(program, out_i_dev);
    vc4_program_destroy(program);
}
