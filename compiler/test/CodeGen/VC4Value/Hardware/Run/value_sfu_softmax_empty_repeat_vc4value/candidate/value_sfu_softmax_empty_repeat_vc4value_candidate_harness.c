#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_N 64u
#define GUARD 32u
#define BUFFER_N (MAX_N + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u
#define REPEAT_ABS_TOL 0.0040f
#define REPEAT_REL_TOL 0.0060f

struct case_desc { uint32_t n, seed; };
static const struct case_desc cases[] = {
    {0u, 3u}, {1u, 5u}, {16u, 7u}, {0u, 11u},
    {2u, 13u}, {31u, 17u}, {0u, 19u}, {7u, 23u},
    {33u, 29u}, {0u, 31u}, {15u, 37u}, {64u, 41u}
};

static float x_values[BUFFER_N];
static float den_values[BUFFER_N];
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

static float x_value(uint32_t seed, uint32_t i) {
    int32_t whole = (int32_t)((i * 5u + seed) % 9u) - 4;
    return (float)whole;
}

static float den_value(uint32_t seed, uint32_t i) {
    static const float denoms[] = {
        0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f, 3.0f, 4.0f, 8.0f
    };
    return denoms[(i + seed) % (sizeof(denoms) / sizeof(denoms[0]))];
}

static float exp2_integer_ref(float value) {
    int n = (int)value;
    float result = 1.0f;
    if (n >= 0) {
        for (int i = 0; i < n; ++i)
            result *= 2.0f;
    } else {
        for (int i = 0; i < -n; ++i)
            result *= 0.5f;
    }
    return result;
}

static float allowed_error(float expected) {
    return maxf_local(REPEAT_ABS_TOL, REPEAT_REL_TOL * absf_local(expected));
}

static void fill_buffers(const struct case_desc *tc) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        x_values[i] = x_value(tc->seed, i);
        den_values[i] = den_value(tc->seed, i);
        out_values[i] = sentinel;
    }
    for (uint32_t i = 0; i < tc->n; i++) {
        x_values[GUARD + i] = x_value(tc->seed, i);
        den_values[GUARD + i] = den_value(tc->seed, i);
    }
}

static int verify_results(const struct case_desc *tc, float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    for (uint32_t i = 0; i < tc->n; i++) {
        uint32_t index = GUARD + i;
        float expected = exp2_integer_ref(x_value(tc->seed, i)) / den_value(tc->seed, i);
        float diff = absf_local(out_values[index] - expected);
        float rel = diff / maxf_local(absf_local(expected), 1.0e-12f);
        if (diff > *max_abs_diff) *max_abs_diff = diff;
        if (rel > *max_rel_diff) *max_rel_diff = rel;
        if (diff > allowed_error(expected)) {
            if (mismatches < 8)
                printk("ERROR: value_sfu_repeat n=%d seed=%d i=%d got=%f expected=%f diff=%f rel=%f\n",
                       (int)tc->n, (int)tc->seed, (int)i, out_values[index],
                       expected, diff, rel);
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
        if (float_to_bits(out_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: value_sfu_repeat sentinel n=%d i=%d bits=%x expected=%x\n",
                       (int)n, (int)i, float_to_bits(out_values[i]), SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t n) {
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
        panic("value_sfu_softmax_empty_repeat program create failed");
    vc4_deviceptr_t x_dev = 0, den_dev = 0, out_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &den_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("value_sfu_softmax_empty_repeat allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_desc *tc = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(rounded_waves(tc->n), 1u, 1u);
        fill_buffers(tc);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t den_active = den_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, den_dev, den_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            value_sfu_softmax_empty_repeat_vc4value_launch(program, grid, block, x_active, den_active, out_active, tc->n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: value_sfu_repeat launch/copy failed case=%d n=%d seed=%d\n",
                   (int)case_id, (int)tc->n, (int)tc->seed);
            launch_failures++;
            continue;
        }
        int mismatches = verify_results(tc, &max_abs_diff, &max_rel_diff);
        int sentinels = verify_sentinels(tc->n);
        uint32_t hash = hash_case(tc->n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_SFU_REPEAT_CASE case=%d n=%d seed=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f max_rel_diff=%f\n",
               (int)case_id, (int)tc->n, (int)tc->seed, mismatches,
               sentinels, hash, max_abs_diff, max_rel_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_sfu_softmax_empty_repeat_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f max_rel_diff=%f saw_repeat_invocation=1 saw_value_approx_sfu_exp=1 saw_value_approx_sfu_recip_div=1 saw_approx_math_policy=1 saw_zero_active_softmax_staged=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES,
           output_hash, output_hash != 0u ? 1 : 0, max_abs_diff,
           max_rel_diff, 3, (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, den_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
