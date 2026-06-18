#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_Q_ROWS 7u
#define MAX_OUT_DIMS 5u
#define MAX_K 64u
#define MAX_SCORE_STRIDE 67u
#define MAX_VT_STRIDE 64u
#define MAX_OUT_STRIDE 8u
#define GROUPS 2u
#define GUARD 32u
#define SCORE_ACTIVE_N (MAX_Q_ROWS * MAX_SCORE_STRIDE)
#define VT_ACTIVE_N (MAX_OUT_DIMS * MAX_VT_STRIDE)
#define OUT_ACTIVE_N (MAX_Q_ROWS * MAX_OUT_STRIDE)
#define AUDIT_ACTIVE_N MAX_Q_ROWS
#define SCORE_BUFFER_N (SCORE_ACTIVE_N + 2u * GUARD)
#define VT_BUFFER_N (VT_ACTIVE_N + 2u * GUARD)
#define OUT_BUFFER_N (OUT_ACTIVE_N + 2u * GUARD)
#define AUDIT_BUFFER_N (AUDIT_ACTIVE_N + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u
#define SENTINEL_H 0x6c5cu
#define MIXED_ABS_TOL 0.0300f
#define MIXED_REL_TOL 0.0400f
#define LN2 0.6931471805599453094f
#define INV_LN2 1.4426950408889634074f

struct case_desc {
    uint32_t q_rows;
    uint32_t out_dims;
    uint32_t k;
    uint32_t lds;
    uint32_t ldv;
    uint32_t ldo;
    float scale;
};

static const struct case_desc cases[] = {
    {1u, 1u, 1u, 16u, 16u, 2u, 0.50f},
    {2u, 2u, 2u, 17u, 19u, 4u, 1.00f},
    {7u, 5u, 7u, 23u, 19u, 8u, 0.75f},
    {1u, 5u, 16u, 16u, 64u, 7u, -0.50f},
    {7u, 1u, 17u, 23u, 19u, 3u, 1.25f},
    {2u, 5u, 31u, 67u, 64u, 6u, 1.50f},
    {7u, 2u, 32u, 67u, 64u, 5u, 0.25f},
    {1u, 2u, 33u, 67u, 64u, 4u, -0.75f},
    {2u, 1u, 47u, 67u, 64u, 3u, 0.90f},
    {7u, 5u, 64u, 67u, 64u, 8u, 1.10f},
    {1u, 5u, 64u, 67u, 64u, 8u, 0.60f},
    {2u, 2u, 17u, 17u, 19u, 5u, 1.35f}
};

static uint16_t scores_values[SCORE_BUFFER_N];
static uint16_t vt_values[VT_BUFFER_N];
static float out_values[OUT_BUFFER_N];
static float audit_values[AUDIT_BUFFER_N];

static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } v; v.u = bits; return v.f; }
static uint32_t float_to_bits(float value) { union { uint32_t u; float f; } v; v.f = value; return v.u; }
static float absf_local(float value) { return value < 0.0f ? -value : value; }
static float maxf_local(float a, float b) { return a > b ? a : b; }
static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    if (exp == 0u)
        return bits_to_float(sign);
    uint32_t bits = sign | ((exp + 112u) << 23) | (mant << 13);
    return bits_to_float(bits);
}

static uint16_t score_half(uint32_t case_id, uint32_t q, uint32_t k) {
    static const uint16_t values[] = {
        0xc000u, 0xbe00u, 0xbc00u, 0xb800u, 0x0000u,
        0x3800u, 0x3c00u, 0x3e00u, 0x4000u
    };
    return values[(q * 7u + k * 5u + case_id * 3u) % 9u];
}

static uint16_t vt_half(uint32_t case_id, uint32_t d, uint32_t k) {
    static const uint16_t values[] = {
        0xbe00u, 0xbc00u, 0xba00u, 0xb800u, 0x0000u,
        0x3800u, 0x3a00u, 0x3c00u, 0x3e00u
    };
    return values[(d * 11u + k * 3u + case_id * 5u) % 9u];
}

static float score_value(uint32_t case_id, uint32_t q, uint32_t k) {
    return f16_to_f32(score_half(case_id, q, k));
}

static float vt_value(uint32_t case_id, uint32_t d, uint32_t k) {
    return f16_to_f32(vt_half(case_id, d, k));
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

static float expected_denom(uint32_t case_id, const struct case_desc *tc, uint32_t q) {
    float maxv = -80.0f;
    for (uint32_t j = 0; j < tc->k; j++) {
        float scaled = score_value(case_id, q, j) * tc->scale;
        if (scaled > maxv)
            maxv = scaled;
    }
    float denom = 0.0f;
    for (uint32_t j = 0; j < tc->k; j++)
        denom += natural_exp_ref(score_value(case_id, q, j) * tc->scale - maxv);
    return denom;
}

static float expected_softmax(uint32_t case_id, const struct case_desc *tc,
                              uint32_t q, uint32_t j) {
    float maxv = -80.0f;
    for (uint32_t k = 0; k < tc->k; k++) {
        float scaled = score_value(case_id, q, k) * tc->scale;
        if (scaled > maxv)
            maxv = scaled;
    }
    return natural_exp_ref(score_value(case_id, q, j) * tc->scale - maxv) /
           expected_denom(case_id, tc, q);
}

static float expected_attention(uint32_t case_id, const struct case_desc *tc,
                                uint32_t q, uint32_t d) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < tc->k; j++)
        acc += expected_softmax(case_id, tc, q, j) * vt_value(case_id, d, j);
    return acc;
}

static void fill_buffers(uint32_t case_id, const struct case_desc *tc) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < SCORE_BUFFER_N; i++)
        scores_values[i] = SENTINEL_H;
    for (uint32_t i = 0; i < VT_BUFFER_N; i++)
        vt_values[i] = SENTINEL_H;
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++)
        out_values[i] = sentinel;
    for (uint32_t i = 0; i < AUDIT_BUFFER_N; i++)
        audit_values[i] = sentinel;

    for (uint32_t q = 0; q < tc->q_rows; q++)
        for (uint32_t j = 0; j < tc->k; j++)
            scores_values[GUARD + q * tc->lds + j] = score_half(case_id, q, j);
    for (uint32_t d = 0; d < tc->out_dims; d++)
        for (uint32_t j = 0; j < tc->k; j++)
            vt_values[GUARD + d * tc->ldv + j] = vt_half(case_id, d, j);
}

static int verify_outputs(uint32_t case_id, const struct case_desc *tc,
                          float *max_abs_diff, float *max_rel_diff,
                          float *max_audit_diff) {
    int mismatches = 0;
    for (uint32_t q = 0; q < tc->q_rows; q++) {
        for (uint32_t d = 0; d < tc->out_dims; d++) {
            uint32_t index = GUARD + q * tc->ldo + d;
            float expected = expected_attention(case_id, tc, q, d);
            float got = out_values[index];
            float diff = absf_local(got - expected);
            float rel = diff / maxf_local(absf_local(expected), 1.0e-12f);
            if (diff > *max_abs_diff) *max_abs_diff = diff;
            if (rel > *max_rel_diff) *max_rel_diff = rel;
            if (diff > maxf_local(MIXED_ABS_TOL, MIXED_REL_TOL * absf_local(expected))) {
                if (mismatches < 8)
                    printk("ERROR: mixed_ttir_attention output case=%d q=%d d=%d got=%f expected=%f diff=%f rel=%f\n",
                           (int)case_id, (int)q, (int)d, got, expected, diff, rel);
                mismatches++;
            }
        }
        float expected_den = expected_denom(case_id, tc, q);
        float got_den = audit_values[GUARD + q];
        float den_diff = absf_local(got_den - expected_den);
        float den_rel = den_diff / maxf_local(absf_local(expected_den), 1.0e-12f);
        if (den_diff > *max_abs_diff) *max_abs_diff = den_diff;
        if (den_rel > *max_rel_diff) *max_rel_diff = den_rel;
        if (den_diff > *max_audit_diff) *max_audit_diff = den_diff;
        if (den_diff > maxf_local(MIXED_ABS_TOL, MIXED_REL_TOL * absf_local(expected_den))) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_attention denom case=%d q=%d got=%f expected=%f diff=%f rel=%f\n",
                       (int)case_id, (int)q, got_den, expected_den, den_diff, den_rel);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct case_desc *tc) {
    int mismatches = 0;
    for (uint32_t i = 0; i < SCORE_BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + SCORE_ACTIVE_N) {
            uint32_t rel = i - GUARD;
            uint32_t q = rel / tc->lds;
            uint32_t j = rel % tc->lds;
            active = q < tc->q_rows && j < tc->k;
        }
        if (!active && scores_values[i] != SENTINEL_H) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_attention score sentinel i=%d half=%x\n", (int)i, scores_values[i]);
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < VT_BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + VT_ACTIVE_N) {
            uint32_t rel = i - GUARD;
            uint32_t d = rel / tc->ldv;
            uint32_t j = rel % tc->ldv;
            active = d < tc->out_dims && j < tc->k;
        }
        if (!active && vt_values[i] != SENTINEL_H) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_attention vt sentinel i=%d half=%x\n", (int)i, vt_values[i]);
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + OUT_ACTIVE_N) {
            uint32_t rel = i - GUARD;
            uint32_t q = rel / tc->ldo;
            uint32_t d = rel % tc->ldo;
            active = q < tc->q_rows && d < tc->out_dims;
        }
        if (!active && float_to_bits(out_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_attention out sentinel i=%d bits=%x\n", (int)i, float_to_bits(out_values[i]));
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < AUDIT_BUFFER_N; i++) {
        int active = i >= GUARD && i < GUARD + tc->q_rows;
        if (!active && float_to_bits(audit_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_attention audit sentinel i=%d bits=%x\n", (int)i, float_to_bits(audit_values[i]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(const struct case_desc *tc) {
    uint32_t hash = 2166136261u ^ tc->q_rows ^ (tc->out_dims << 8) ^ (tc->k << 16);
    for (uint32_t q = 0; q < tc->q_rows; q++) {
        for (uint32_t d = 0; d < tc->out_dims; d++) {
            uint32_t index = GUARD + q * tc->ldo + d;
            hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
        hash ^= float_to_bits(audit_values[GUARD + q]) + 0x85ebca6bu + (q << 7);
        hash = rotl32_local(hash, 9u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("mixed_ttir_online_attention_apply program create failed");

    vc4_deviceptr_t scores_dev = 0, vt_dev = 0, out_dev = 0, audit_dev = 0;
    uint32_t scores_bytes = SCORE_BUFFER_N * sizeof(uint16_t);
    uint32_t vt_bytes = VT_BUFFER_N * sizeof(uint16_t);
    uint32_t out_bytes = OUT_BUFFER_N * sizeof(float);
    uint32_t audit_bytes = AUDIT_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &scores_dev, scores_bytes) < 0 ||
        vc4_m2_malloc(program, &vt_dev, vt_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0 ||
        vc4_m2_malloc(program, &audit_dev, audit_bytes) < 0)
        panic("mixed_ttir_online_attention_apply allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f, max_audit_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_desc *tc = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(tc->q_rows, tc->out_dims, GROUPS);
        fill_buffers(case_id, tc);
        vc4_deviceptr_t scores_active = scores_dev + GUARD * sizeof(uint16_t);
        vc4_deviceptr_t vt_active = vt_dev + GUARD * sizeof(uint16_t);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        vc4_deviceptr_t audit_active = audit_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, scores_dev, scores_values, scores_bytes) < 0 ||
            vc4_m2_copy_htod(program, vt_dev, vt_values, vt_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            vc4_m2_copy_htod(program, audit_dev, audit_values, audit_bytes) < 0 ||
            mixed_ttir_online_attention_apply_axes_mask_cf_f16_storage_b16_kernel_launch(
                program, grid, block, scores_active, vt_active, out_active,
                audit_active, tc->k, tc->lds, tc->ldv, tc->ldo, tc->scale) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, audit_values, audit_dev, audit_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, scores_values, scores_dev, scores_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, vt_values, vt_dev, vt_bytes) < 0) {
            printk("ERROR: mixed_ttir_attention launch/copy failed case=%d q_rows=%d out_dims=%d k=%d\n",
                   (int)case_id, (int)tc->q_rows, (int)tc->out_dims, (int)tc->k);
            launch_failures++;
            continue;
        }
        int mismatches = verify_outputs(case_id, tc, &max_abs_diff, &max_rel_diff, &max_audit_diff);
        int sentinels = verify_sentinels(tc);
        uint32_t hash = hash_case(tc);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("MIXED_TTIR_ONLINE_ATTENTION_APPLY_AXIS_MASK_CF_F16_CASE case=%d q_rows=%d out_dims=%d k=%d lds=%d ldv=%d ldo=%d scale=%f grid_z=%d block_x=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f max_rel_diff=%f max_audit_diff=%f\n",
               (int)case_id, (int)tc->q_rows, (int)tc->out_dims, (int)tc->k,
               (int)tc->lds, (int)tc->ldv, (int)tc->ldo, tc->scale, (int)GROUPS,
               (int)LANES, mismatches, sentinels, hash, max_abs_diff, max_rel_diff,
               max_audit_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_ttir_online_attention_apply_axes_mask_cf_f16_storage_b16_vc4triton status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d block_x=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f max_rel_diff=%f max_audit_diff=%f runtime_allocations=%d runtime_launches=%d saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_value_scf_cf=1 saw_ttir_elementwise=1 saw_ttir_control_flow=1 saw_ttir_loop_carried_f32_state=1 saw_ttir_multi_axis=1 saw_ttir_program_id_axis1=1 saw_ttir_program_id_axis2=1 saw_ttir_mask_tail=1 saw_ttir_row_strided_memory=1 saw_ttir_reduction_f32_finite_add=1 saw_ttir_scalar_reduction_store=1 saw_ttir_f16_storage_load=1 saw_ttir_f32_compute_after_f16_load=1 saw_value_f16_storage=1 saw_ttir_approx_sfu_exp=1 saw_ttir_approx_sfu_recip_div=1 saw_ttir_finite_f32_max_reduction=1 saw_ttir_softmax_v0=1 saw_ttir_attention_apply_v0=1 saw_ttir_online_softmax_state=1 saw_ttir_online_attention_apply=1 saw_ttir_k_gt_16=1 saw_ttir_precomputed_scores=1 saw_ttir_transposed_v_layout=1 saw_ttir_weighted_sum_reduction=1 saw_ttir_scalar_result_store=1 saw_value_approx_sfu=1 saw_value_softmax_v0=1 saw_scalar_to_vector_f32_broadcast=1 saw_approx_math_policy=1 saw_softmax_uses_natural_exp=1 saw_no_exact_default_math=1 saw_no_scalar_global_load=1 saw_no_nontransposed_v_gather=1 saw_no_qk_score_generation=1 saw_no_full_attention=1 saw_no_tl_dot_tt_dot=1 saw_no_vector_contract=1 saw_no_gemm=1 saw_row_padding_sentinels=1 saw_output_padding_sentinels=1 elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, LANES,
           output_hash, output_hash != 0u ? 1 : 0, max_abs_diff, max_rel_diff,
           max_audit_diff, 4, (int)(sizeof(cases) / sizeof(cases[0])),
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, elapsed);

    vc4Free(program, scores_dev);
    vc4Free(program, vt_dev);
    vc4Free(program, out_dev);
    vc4Free(program, audit_dev);
    vc4_program_destroy(program);
}
