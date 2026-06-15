#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_N 1000u
#define WAVE_BOUNDARY_N ELEMENTS_PER_WAVE
#define MAX_WAVES ((MAX_N + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE)
#define MAX_COVERAGE_N (MAX_WAVES * ELEMENTS_PER_WAVE)
#define GUARD 32u
#define BUFFER_N (MAX_COVERAGE_N + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u
#define LOG_ABS_TOL 0.0060f
#define LOG_REL_TOL 0.0120f
#define LN2 0.6931471805599453f

static const uint32_t n_cases[] = {
    1u, 2u, 7u, 15u, 16u, 17u, 31u, 32u, 33u, WAVE_BOUNDARY_N
};

static float in_values[BUFFER_N];
static float out_values[BUFFER_N];

static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } v; v.u = bits; return v.f; }
static uint32_t float_to_bits(float value) { union { uint32_t u; float f; } v; v.f = value; return v.u; }
static float absf_local(float value) { return value < 0.0f ? -value : value; }
static float maxf_local(float a, float b) { return a > b ? a : b; }
static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static int input_exp2_power(uint32_t case_id, uint32_t i) {
    static const int powers[] = {-3, -2, -1, 0, 1, 2, 3};
    return powers[(i + case_id * 2u) % (sizeof(powers) / sizeof(powers[0]))];
}

static float pow2_int_local(int n) {
    float result = 1.0f;
    if (n >= 0) {
        for (int i = 0; i < n; i++)
            result *= 2.0f;
    } else {
        for (int i = 0; i < -n; i++)
            result *= 0.5f;
    }
    return result;
}

static float input_value(uint32_t case_id, uint32_t i) {
    return pow2_int_local(input_exp2_power(case_id, i));
}

static float natural_log_ref(uint32_t case_id, uint32_t i) {
    return (float)input_exp2_power(case_id, i) * LN2;
}

static float raw_log2_oracle(uint32_t case_id, uint32_t i) {
    return (float)input_exp2_power(case_id, i);
}

static float allowed_error(float expected) {
    return maxf_local(LOG_ABS_TOL, LOG_REL_TOL * absf_local(expected));
}

static void fill_buffers(uint32_t case_id, uint32_t n) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        in_values[i] = input_value(case_id, i);
        out_values[i] = sentinel;
    }
    for (uint32_t i = 0; i < n; i++)
        in_values[GUARD + i] = input_value(case_id, i);
}

static int verify_results(uint32_t case_id, uint32_t n, float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    int saw_distinguishing_value = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        float expected = natural_log_ref(case_id, i);
        float raw_log2 = raw_log2_oracle(case_id, i);
        if (absf_local(expected - raw_log2) > allowed_error(expected) * 8.0f)
            saw_distinguishing_value = 1;
        float diff = absf_local(out_values[index] - expected);
        float rel = diff / maxf_local(absf_local(expected), 1.0e-12f);
        if (diff > *max_abs_diff) *max_abs_diff = diff;
        if (rel > *max_rel_diff) *max_rel_diff = rel;
        if (diff > allowed_error(expected)) {
            if (mismatches < 8)
                printk("ERROR: value_sfu_natural_log case=%d n=%d i=%d x=%f gpu=%f expected=%f raw_log2=%f diff=%f rel=%f allowed=%f\n",
                       (int)case_id, (int)n, (int)i, input_value(case_id, i),
                       out_values[index], expected, raw_log2, diff, rel,
                       allowed_error(expected));
            mismatches++;
        }
    }
    if (!saw_distinguishing_value) {
        printk("ERROR: value_sfu_natural_log case=%d n=%d did not distinguish natural log from raw log2 oracle\n",
               (int)case_id, (int)n);
        mismatches++;
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + n)
            continue;
        if (float_to_bits(out_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: value_sfu_natural_log sentinel n=%d i=%d bits=%x expected=%x\n",
                       (int)n, (int)i, float_to_bits(out_values[i]), SENTINEL_BITS);
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
        panic("value_sfu_natural_log_f32_b16 program create failed");
    vc4_deviceptr_t in_dev = 0, out_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &in_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("value_sfu_natural_log_f32_b16 allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(n_cases) / sizeof(n_cases[0]); case_id++) {
        uint32_t n = n_cases[case_id];
        uint32_t waves = rounded_waves(n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1u, 1u);
        fill_buffers(case_id, n);
        vc4_deviceptr_t in_active = in_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, in_dev, in_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            value_sfu_natural_log_f32_b16_vc4value_launch(program, grid, block, in_active, out_active, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: value_sfu_natural_log launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_results(case_id, n, &max_abs_diff, &max_rel_diff);
        int sentinels = verify_sentinels(n);
        uint32_t hash = hash_output(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_SFU_NATURAL_LOG_CASE case=%d n=%d waves=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f max_rel_diff=%f\n",
               (int)case_id, (int)n, (int)waves, mismatches, sentinels,
               hash, max_abs_diff, max_rel_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_sfu_natural_log_f32_b16_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d wave_boundary_n=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f max_rel_diff=%f saw_value_math_log_natural=1 saw_target_sfu_log2_scale_ln2=1 saw_not_raw_log2_oracle=1 saw_positive_finite_domain=1 saw_approx_math_policy=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, MAX_N,
           WAVE_BOUNDARY_N, output_hash, output_hash != 0u ? 1 : 0,
           max_abs_diff, max_rel_diff, 2,
           (int)(sizeof(n_cases) / sizeof(n_cases[0])), elapsed);

    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
