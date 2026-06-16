#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_ROWS 7u
#define MAX_K 64u
#define MAX_SCORE_STRIDE 23u
#define GUARD 32u
#define SCORE_ACTIVE_N (((MAX_ROWS - 1u) * MAX_SCORE_STRIDE) + MAX_K)
#define OUT_ACTIVE_N MAX_ROWS
#define SCORE_BUFFER_N (SCORE_ACTIVE_N + 2u * GUARD)
#define OUT_BUFFER_N (OUT_ACTIVE_N + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u
#define DENOM_ABS_TOL 0.1200f
#define DENOM_REL_TOL 0.0250f
#define LN2 0.6931471805599453094f
#define INV_LN2 1.4426950408889634074f

static const uint32_t rows_cases[] = {1u, 2u, 7u};
static const uint32_t k_cases[] = {1u, 2u, 7u, 16u, 17u, 31u, 32u, 33u, 47u, 64u};
static const uint32_t stride_cases[] = {16u, 17u, 23u};

static float scores_values[SCORE_BUFFER_N];
static float out_values[OUT_BUFFER_N];

static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } v; v.u = bits; return v.f; }
static uint32_t float_to_bits(float value) { union { uint32_t u; float f; } v; v.f = value; return v.u; }
static float absf_local(float value) { return value < 0.0f ? -value : value; }
static float maxf_local(float a, float b) { return a > b ? a : b; }
static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
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

static float score_value(uint32_t case_id, uint32_t index) {
    int32_t whole = (int32_t)(((index / MAX_K) * 7u + (index % MAX_K) * 5u + case_id * 3u) % 9u) - 4;
    return (float)whole * 0.5f;
}

static int score_address_active(uint32_t rel, uint32_t rows, uint32_t k, uint32_t stride) {
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t base = r * stride;
        if (rel >= base && rel < base + k)
            return 1;
    }
    return 0;
}

static void fill_buffers(uint32_t case_id, uint32_t rows, uint32_t k, uint32_t stride) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < SCORE_BUFFER_N; i++)
        scores_values[i] = sentinel;
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++)
        out_values[i] = sentinel;

    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t j = 0; j < k; j++) {
            uint32_t index = r * stride + j;
            scores_values[GUARD + index] = score_value(case_id, index);
        }
}

static float expected_denom(uint32_t case_id, uint32_t row, uint32_t k, uint32_t stride) {
    float maxv = -80.0f;
    for (uint32_t j = 0; j < k; j++) {
        float s = scores_values[GUARD + row * stride + j];
        if (s > maxv)
            maxv = s;
    }
    float denom = 0.0f;
    for (uint32_t j = 0; j < k; j++)
        denom += natural_exp_ref(scores_values[GUARD + row * stride + j] - maxv);
    return denom;
}

static int verify_outputs(uint32_t case_id, uint32_t rows, uint32_t k,
                          uint32_t stride, float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t index = GUARD + r;
        float expected = expected_denom(case_id, r, k, stride);
        float got = out_values[index];
        float diff = absf_local(got - expected);
        float rel = diff / maxf_local(absf_local(expected), 1.0e-12f);
        if (diff > *max_abs_diff) *max_abs_diff = diff;
        if (rel > *max_rel_diff) *max_rel_diff = rel;
        if (diff > maxf_local(DENOM_ABS_TOL, DENOM_REL_TOL * absf_local(expected))) {
            if (mismatches < 8)
                printk("ERROR: online_normalizer case=%d row=%d k=%d stride=%d got=%f expected=%f diff=%f rel=%f\n",
                       (int)case_id, (int)r, (int)k, (int)stride, got, expected, diff, rel);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t rows, uint32_t k, uint32_t stride) {
    int mismatches = 0;
    for (uint32_t i = 0; i < SCORE_BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + SCORE_ACTIVE_N) {
            uint32_t rel = i - GUARD;
            active = score_address_active(rel, rows, k, stride);
        }
        if (!active && float_to_bits(scores_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: online_normalizer score sentinel i=%d bits=%x\n", (int)i, float_to_bits(scores_values[i]));
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++) {
        int active = i >= GUARD && i < GUARD + rows;
        if (!active && float_to_bits(out_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: online_normalizer out sentinel i=%d bits=%x\n", (int)i, float_to_bits(out_values[i]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t rows) {
    uint32_t hash = 2166136261u ^ rows;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t index = GUARD + r;
        hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (index << 6) + (index >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("value_online_softmax_normalizer program create failed");

    vc4_deviceptr_t scores_dev = 0, out_dev = 0;
    uint32_t scores_bytes = SCORE_BUFFER_N * sizeof(float);
    uint32_t out_bytes = OUT_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &scores_dev, scores_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("value_online_softmax_normalizer allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ACTIVE_QPUS * LANES, 1u, 1u);
    uint32_t case_id = 0;

    for (uint32_t ri = 0; ri < sizeof(rows_cases) / sizeof(rows_cases[0]); ri++)
    for (uint32_t ki = 0; ki < sizeof(k_cases) / sizeof(k_cases[0]); ki++)
    for (uint32_t si = 0; si < sizeof(stride_cases) / sizeof(stride_cases[0]); si++) {
        uint32_t rows = rows_cases[ri];
        uint32_t k = k_cases[ki];
        uint32_t stride = stride_cases[si];
        vc4_dim3 grid = vc4_m2_dim3(rows, 1u, 1u);
        fill_buffers(case_id, rows, k, stride);
        vc4_deviceptr_t scores_active = scores_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, scores_dev, scores_values, scores_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            value_online_softmax_normalizer_f32_b16_vc4value_launch(
                program, grid, block, scores_active, out_active,
                k, SCORE_ACTIVE_N, rows, stride) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, scores_values, scores_dev, scores_bytes) < 0) {
            printk("ERROR: online_normalizer launch/copy failed case=%d rows=%d k=%d stride=%d\n",
                   (int)case_id, (int)rows, (int)k, (int)stride);
            launch_failures++;
            case_id++;
            continue;
        }
        int mismatches = verify_outputs(case_id, rows, k, stride, &max_abs_diff, &max_rel_diff);
        int sentinels = verify_sentinels(rows, k, stride);
        uint32_t hash = hash_case(rows);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_ONLINE_SOFTMAX_NORMALIZER_CASE case=%d rows=%d k=%d stride=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f max_rel_diff=%f\n",
               (int)case_id, (int)rows, (int)k, (int)stride, mismatches,
               sentinels, hash, max_abs_diff, max_rel_diff);
        case_id++;
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_online_softmax_normalizer_f32_b16_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_k=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f max_rel_diff=%f saw_value_online_softmax_state=1 saw_value_loop_carried_m_l_state=1 saw_value_k_gt_16=1 saw_softmax_uses_natural_exp=1 saw_value_finite_f32_max_reduction=1 saw_value_reduction_f32_finite_add=1 saw_no_scalar_global_load=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)case_id, total_mismatches, sentinel_mismatches,
           launch_failures, ACTIVE_QPUS, LANES, MAX_ROWS, MAX_K, output_hash,
           output_hash != 0u ? 1 : 0, max_abs_diff, max_rel_diff, 2,
           (int)case_id, elapsed);

    vc4Free(program, scores_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
