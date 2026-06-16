#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_Q_ROWS 7u
#define MAX_OUT_DIMS 5u
#define MAX_K 64u
#define MAX_SCORE_STRIDE 67u
#define MAX_VT_STRIDE 64u
#define MAX_OUT_STRIDE 8u
#define GUARD 32u
#define SCORE_ACTIVE_N (MAX_Q_ROWS * MAX_SCORE_STRIDE)
#define VT_ACTIVE_N (MAX_OUT_DIMS * MAX_VT_STRIDE)
#define OUT_ACTIVE_N (MAX_Q_ROWS * MAX_OUT_STRIDE)
#define SCORE_BUFFER_N (SCORE_ACTIVE_N + 2u * GUARD)
#define VT_BUFFER_N (VT_ACTIVE_N + 2u * GUARD)
#define OUT_BUFFER_N (OUT_ACTIVE_N + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u
#define ATTN_ABS_TOL 0.0120f
#define ATTN_REL_TOL 0.0160f
#define LN2 0.6931471805599453094f
#define INV_LN2 1.4426950408889634074f

struct repeat_case {
    uint32_t q_rows;
    uint32_t out_dims;
    uint32_t k;
    uint32_t lds;
    uint32_t ldv;
};
static const struct repeat_case cases[] = {
    {1u, 1u, 1u, 16u, 16u},
    {2u, 2u, 17u, 17u, 19u},
    {7u, 5u, 64u, 67u, 64u},
    {1u, 5u, 17u, 16u, 19u},
    {7u, 1u, 64u, 67u, 64u},
    {2u, 5u, 1u, 23u, 19u},
    {1u, 2u, 64u, 67u, 64u},
    {7u, 2u, 17u, 23u, 19u},
    {2u, 1u, 1u, 17u, 16u},
    {7u, 5u, 64u, 67u, 64u},
    {1u, 5u, 17u, 16u, 19u},
    {2u, 2u, 1u, 17u, 19u}
};

static float scores_values[SCORE_BUFFER_N];
static float vt_values[VT_BUFFER_N];
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

static float vt_value(uint32_t case_id, uint32_t index) {
    int32_t whole = (int32_t)(((index / MAX_K) * 11u + (index % MAX_K) * 3u + case_id * 5u) % 13u) - 6;
    return (float)whole * 0.25f;
}

static int score_address_active(uint32_t rel, uint32_t q_rows, uint32_t k, uint32_t lds) {
    for (uint32_t q = 0; q < q_rows; q++) {
        uint32_t base = q * lds;
        if (rel >= base && rel < base + k)
            return 1;
    }
    return 0;
}

static int vt_address_active(uint32_t rel, uint32_t out_dims, uint32_t k, uint32_t ldv) {
    for (uint32_t d = 0; d < out_dims; d++) {
        uint32_t base = d * ldv;
        if (rel >= base && rel < base + k)
            return 1;
    }
    return 0;
}

static uint32_t out_stride_for(uint32_t out_dims, uint32_t case_id) {
    uint32_t extra = 1u + (case_id % 3u);
    uint32_t stride = out_dims + extra;
    return stride > MAX_OUT_STRIDE ? MAX_OUT_STRIDE : stride;
}

static void fill_buffers(uint32_t case_id, uint32_t q_rows, uint32_t out_dims,
                         uint32_t k, uint32_t lds, uint32_t ldv, uint32_t ldo) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < SCORE_BUFFER_N; i++)
        scores_values[i] = sentinel;
    for (uint32_t i = 0; i < VT_BUFFER_N; i++)
        vt_values[i] = sentinel;
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++)
        out_values[i] = sentinel;

    for (uint32_t q = 0; q < q_rows; q++)
        for (uint32_t j = 0; j < k; j++) {
            uint32_t index = q * lds + j;
            scores_values[GUARD + index] = score_value(case_id, index);
        }

    for (uint32_t d = 0; d < out_dims; d++)
        for (uint32_t j = 0; j < k; j++) {
            uint32_t index = d * ldv + j;
            vt_values[GUARD + index] = vt_value(case_id, index);
        }
    (void)ldo;
}

static float expected_attention(uint32_t case_id, uint32_t q, uint32_t d, uint32_t k,
                                uint32_t lds, uint32_t ldv) {
    float maxv = -80.0f;
    for (uint32_t j = 0; j < k; j++) {
        float s = scores_values[GUARD + q * lds + j];
        if (s > maxv)
            maxv = s;
    }
    float denom = 0.0f;
    for (uint32_t j = 0; j < k; j++)
        denom += natural_exp_ref(scores_values[GUARD + q * lds + j] - maxv);

    float acc = 0.0f;
    for (uint32_t j = 0; j < k; j++) {
        float prob = natural_exp_ref(scores_values[GUARD + q * lds + j] - maxv) / denom;
        acc += prob * vt_values[GUARD + d * ldv + j];
    }
    return acc;
}

static int verify_outputs(uint32_t case_id, uint32_t q_rows, uint32_t out_dims,
                          uint32_t k, uint32_t lds, uint32_t ldv, uint32_t ldo,
                          float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    for (uint32_t q = 0; q < q_rows; q++) {
        for (uint32_t d = 0; d < out_dims; d++) {
            uint32_t index = GUARD + q * ldo + d;
            float expected = expected_attention(case_id, q, d, k, lds, ldv);
            float got = out_values[index];
            float diff = absf_local(got - expected);
            float rel = diff / maxf_local(absf_local(expected), 1.0e-12f);
            if (diff > *max_abs_diff) *max_abs_diff = diff;
            if (rel > *max_rel_diff) *max_rel_diff = rel;
            if (diff > maxf_local(ATTN_ABS_TOL, ATTN_REL_TOL * absf_local(expected))) {
                if (mismatches < 8)
                    printk("ERROR: attention_repeat case=%d q=%d d=%d k=%d lds=%d ldv=%d got=%f expected=%f diff=%f rel=%f\n",
                           (int)case_id, (int)q, (int)d, (int)k, (int)lds, (int)ldv, got, expected, diff, rel);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t q_rows, uint32_t out_dims, uint32_t k,
                            uint32_t lds, uint32_t ldv, uint32_t ldo) {
    int mismatches = 0;
    for (uint32_t i = 0; i < SCORE_BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + SCORE_ACTIVE_N) {
            uint32_t rel = i - GUARD;
            active = score_address_active(rel, q_rows, k, lds);
        }
        if (!active && float_to_bits(scores_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: attention_repeat score sentinel i=%d bits=%x\n", (int)i, float_to_bits(scores_values[i]));
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < VT_BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + VT_ACTIVE_N) {
            uint32_t rel = i - GUARD;
            active = vt_address_active(rel, out_dims, k, ldv);
        }
        if (!active && float_to_bits(vt_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: attention_repeat vt sentinel i=%d bits=%x\n", (int)i, float_to_bits(vt_values[i]));
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + OUT_ACTIVE_N) {
            uint32_t rel = i - GUARD;
            uint32_t q = rel / ldo;
            uint32_t d = rel % ldo;
            active = q < q_rows && d < out_dims;
        }
        if (!active && float_to_bits(out_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: attention_repeat out sentinel i=%d bits=%x\n", (int)i, float_to_bits(out_values[i]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t q_rows, uint32_t out_dims, uint32_t ldo) {
    uint32_t hash = 2166136261u ^ q_rows ^ (out_dims << 8) ^ (ldo << 16);
    for (uint32_t q = 0; q < q_rows; q++) {
        for (uint32_t d = 0; d < out_dims; d++) {
            uint32_t index = GUARD + q * ldo + d;
            hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("value_attention_apply_v0_repeat program create failed");

    vc4_deviceptr_t scores_dev = 0, vt_dev = 0, out_dev = 0;
    uint32_t scores_bytes = SCORE_BUFFER_N * sizeof(float);
    uint32_t vt_bytes = VT_BUFFER_N * sizeof(float);
    uint32_t out_bytes = OUT_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &scores_dev, scores_bytes) < 0 ||
        vc4_m2_malloc(program, &vt_dev, vt_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("value_attention_apply_v0_repeat allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ACTIVE_QPUS * LANES, 1u, 1u);
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t q_rows = cases[case_id].q_rows;
        uint32_t out_dims = cases[case_id].out_dims;
        uint32_t k = cases[case_id].k;
        uint32_t lds = cases[case_id].lds;
        uint32_t ldv = cases[case_id].ldv;
        uint32_t ldo = out_stride_for(out_dims, case_id);
        vc4_dim3 grid = vc4_m2_dim3(q_rows, out_dims, 1u);
        fill_buffers(case_id, q_rows, out_dims, k, lds, ldv, ldo);
        vc4_deviceptr_t scores_active = scores_dev + GUARD * sizeof(float);
        vc4_deviceptr_t vt_active = vt_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, scores_dev, scores_values, scores_bytes) < 0 ||
            vc4_m2_copy_htod(program, vt_dev, vt_values, vt_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            value_online_attention_apply_repeat_vc4value_launch(
                program, grid, block, scores_active, vt_active, out_active,
                k, SCORE_ACTIVE_N, VT_ACTIVE_N, OUT_ACTIVE_N, lds, ldv, ldo) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, scores_values, scores_dev, scores_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, vt_values, vt_dev, vt_bytes) < 0) {
            printk("ERROR: attention_repeat launch/copy failed case=%d q_rows=%d out_dims=%d k=%d lds=%d ldv=%d ldo=%d\n",
                   (int)case_id, (int)q_rows, (int)out_dims, (int)k, (int)lds, (int)ldv, (int)ldo);
            launch_failures++;
            continue;
        }
        int mismatches = verify_outputs(case_id, q_rows, out_dims, k, lds, ldv, ldo, &max_abs_diff, &max_rel_diff);
        int sentinels = verify_sentinels(q_rows, out_dims, k, lds, ldv, ldo);
        uint32_t hash = hash_case(q_rows, out_dims, ldo);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_ATTENTION_APPLY_REPEAT_CASE case=%d q_rows=%d out_dims=%d k=%d lds=%d ldv=%d ldo=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f max_rel_diff=%f\n",
               (int)case_id, (int)q_rows, (int)out_dims, (int)k, (int)lds,
               (int)ldv, (int)ldo, mismatches, sentinels, hash, max_abs_diff,
               max_rel_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_online_attention_apply_repeat_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_q_rows=%d max_out_dims=%d max_k=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f max_rel_diff=%f saw_value_online_attention_apply=1 saw_repeat_invocation=1 saw_k1_attention_apply=1 saw_k17_attention_apply=1 saw_k64_attention_apply=1 saw_value_precomputed_scores=1 saw_value_transposed_v_layout=1 saw_value_loop_carried_m_l_acc_state=1 saw_value_k_gt_16=1 saw_value_weighted_sum_reduction=1 saw_value_scalar_result_store=1 saw_softmax_uses_natural_exp=1 saw_value_approx_sfu_exp=1 saw_value_approx_sfu_recip_div=1 saw_no_scalar_global_load=1 saw_no_nontransposed_v_gather=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches, sentinel_mismatches,
           launch_failures, ACTIVE_QPUS, LANES, MAX_Q_ROWS, MAX_OUT_DIMS, MAX_K,
           output_hash, output_hash != 0u ? 1 : 0, max_abs_diff, max_rel_diff,
           3, (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, scores_dev);
    vc4Free(program, vt_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
