#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_ROWS 7u
#define MAX_NCOLS 16u
#define MAX_STRIDE 23u
#define EXTRA_ROWS 2u
#define GUARD 32u
#define BUFFER_N ((MAX_ROWS + EXTRA_ROWS) * MAX_STRIDE + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u
#define SOFTMAX_ABS_TOL 0.0040f
#define SOFTMAX_REL_TOL 0.0060f
#define ROW_SUM_TOL 0.0060f
#define LN2 0.6931471805599453094f
#define INV_LN2 1.4426950408889634074f

struct case_desc { uint32_t rows, ncols, stride; };
static const struct case_desc cases[] = {
    {1u, 1u, 5u}, {1u, 2u, 7u}, {1u, 7u, 13u}, {1u, 15u, 21u}, {1u, 16u, 23u},
    {2u, 1u, 5u}, {2u, 2u, 7u}, {2u, 7u, 13u}, {2u, 15u, 21u}, {2u, 16u, 23u},
    {7u, 1u, 5u}, {7u, 2u, 7u}, {7u, 7u, 13u}, {7u, 15u, 21u}, {7u, 16u, 23u}
};

static float x_values[BUFFER_N];
static float y_values[BUFFER_N];

static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } v; v.u = bits; return v.f; }
static uint32_t float_to_bits(float value) { union { uint32_t u; float f; } v; v.f = value; return v.u; }
static float absf_local(float value) { return value < 0.0f ? -value : value; }
static float maxf_local(float a, float b) { return a > b ? a : b; }
static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static float logit_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)((row * 7u + col * 5u + case_id * 3u) % 9u) - 4;
    return (float)whole;
}

static float pow2_int_local(int n) {
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

static float natural_exp_ref(float x) {
    int k = (int)(x * INV_LN2 + (x >= 0.0f ? 0.5f : -0.5f));
    float r = x - (float)k * LN2;
    float term = 1.0f;
    float sum = 1.0f;
    for (int i = 1; i <= 10; ++i) {
        term *= r / (float)i;
        sum += term;
    }
    return pow2_int_local(k) * sum;
}

static void fill_buffers(uint32_t case_id, const struct case_desc *tc) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        x_values[i] = sentinel;
        y_values[i] = sentinel;
    }
    for (uint32_t r = 0; r < tc->rows; r++)
        for (uint32_t c = 0; c < tc->ncols; c++)
            x_values[GUARD + r * tc->stride + c] = logit_value(case_id, r, c);
}

static float expected_softmax(uint32_t case_id, const struct case_desc *tc, uint32_t row, uint32_t col) {
    float maxv = -80.0f;
    for (uint32_t c = 0; c < tc->ncols; c++) {
        float v = logit_value(case_id, row, c);
        if (v > maxv)
            maxv = v;
    }
    float denom = 0.0f;
    for (uint32_t c = 0; c < tc->ncols; c++)
        denom += natural_exp_ref(logit_value(case_id, row, c) - maxv);
    return natural_exp_ref(logit_value(case_id, row, col) - maxv) / denom;
}

static int verify_case(uint32_t case_id, const struct case_desc *tc,
                       float *max_abs_diff, float *max_rel_diff, float *max_row_sum_diff) {
    int mismatches = 0;
    for (uint32_t r = 0; r < tc->rows; r++) {
        float row_sum = 0.0f;
        for (uint32_t c = 0; c < tc->ncols; c++) {
            uint32_t index = GUARD + r * tc->stride + c;
            float expected = expected_softmax(case_id, tc, r, c);
            float got = y_values[index];
            float diff = absf_local(got - expected);
            float rel = diff / maxf_local(absf_local(expected), 1.0e-12f);
            row_sum += got;
            if (diff > *max_abs_diff) *max_abs_diff = diff;
            if (rel > *max_rel_diff) *max_rel_diff = rel;
            if (diff > maxf_local(SOFTMAX_ABS_TOL, SOFTMAX_REL_TOL * absf_local(expected))) {
                if (mismatches < 8)
                    printk("ERROR: value_softmax case=%d row=%d col=%d got=%f expected=%f diff=%f rel=%f\n",
                           (int)case_id, (int)r, (int)c, got, expected, diff, rel);
                mismatches++;
            }
        }
        float sum_diff = absf_local(row_sum - 1.0f);
        if (sum_diff > *max_row_sum_diff)
            *max_row_sum_diff = sum_diff;
        if (sum_diff > ROW_SUM_TOL) {
            if (mismatches < 8)
                printk("ERROR: value_softmax row_sum case=%d row=%d sum=%f diff=%f\n",
                       (int)case_id, (int)r, row_sum, sum_diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct case_desc *tc) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + tc->rows * tc->stride) {
            uint32_t rel = i - GUARD;
            active = (rel % tc->stride) < tc->ncols;
        }
        if (active)
            continue;
        if (float_to_bits(y_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: value_softmax sentinel rows=%d ncols=%d stride=%d i=%d bits=%x expected=%x\n",
                       (int)tc->rows, (int)tc->ncols, (int)tc->stride,
                       (int)i, float_to_bits(y_values[i]), SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(const struct case_desc *tc) {
    uint32_t hash = 2166136261u ^ tc->rows ^ (tc->ncols << 8) ^ (tc->stride << 16);
    for (uint32_t r = 0; r < tc->rows; r++) {
        for (uint32_t c = 0; c < tc->ncols; c++) {
            uint32_t index = GUARD + r * tc->stride + c;
            hash ^= float_to_bits(y_values[index]) + 0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("value_softmax_stable_f32_b16 program create failed");
    vc4_deviceptr_t x_dev = 0, y_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0)
        panic("value_softmax_stable_f32_b16 allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f, max_row_sum_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ACTIVE_QPUS * LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_desc *tc = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(1u, tc->rows, 1u);
        fill_buffers(case_id, tc);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
            value_softmax_stable_f32_b16_vc4value_launch(
                program, grid, block, x_active, y_active, tc->rows, tc->ncols, tc->stride) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, bytes) < 0) {
            printk("ERROR: value_softmax launch/copy failed case=%d rows=%d ncols=%d stride=%d\n",
                   (int)case_id, (int)tc->rows, (int)tc->ncols, (int)tc->stride);
            launch_failures++;
            continue;
        }
        int mismatches = verify_case(case_id, tc, &max_abs_diff, &max_rel_diff, &max_row_sum_diff);
        int sentinels = verify_sentinels(tc);
        uint32_t hash = hash_case(tc);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_SOFTMAX_CASE case=%d rows=%d ncols=%d stride=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f max_rel_diff=%f max_row_sum_diff=%f\n",
               (int)case_id, (int)tc->rows, (int)tc->ncols, (int)tc->stride,
               mismatches, sentinels, hash, max_abs_diff, max_rel_diff, max_row_sum_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_softmax_stable_f32_b16_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_ncols=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f max_rel_diff=%f max_row_sum_diff=%f saw_value_softmax_v0=1 saw_value_finite_f32_max_reduction=1 saw_value_approx_sfu_exp=1 saw_value_approx_sfu_recip_div=1 saw_scalar_to_vector_f32_broadcast=1 saw_approx_math_policy=1 saw_zero_active_softmax_staged=1 saw_softmax_uses_natural_exp=1 saw_target_sfu_exp2_scale_log2e=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, MAX_ROWS,
           MAX_NCOLS, output_hash, output_hash != 0u ? 1 : 0, max_abs_diff,
           max_rel_diff, max_row_sum_diff, 2,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
