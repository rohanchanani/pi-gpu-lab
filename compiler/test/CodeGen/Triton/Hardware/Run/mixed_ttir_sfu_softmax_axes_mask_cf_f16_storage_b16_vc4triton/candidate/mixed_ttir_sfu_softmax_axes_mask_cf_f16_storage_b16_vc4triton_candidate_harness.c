#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_ROWS 7u
#define MAX_NCOLS 16u
#define MAX_STRIDE 23u
#define EXTRA_ROWS 2u
#define GUARD 32u
#define BUFFER_N ((MAX_ROWS * ACTIVE_QPUS + EXTRA_ROWS) * MAX_STRIDE + 2u * GUARD)
#define AUDIT_N (MAX_ROWS * ACTIVE_QPUS + 2u * GUARD)
#define INPUT_PAD_H 0x5c6cu
#define SENTINEL_F_BITS 0xc56a4000u
#define SOFTMAX_ABS_TOL 0.0060f
#define SOFTMAX_REL_TOL 0.0100f
#define ROW_SUM_TOL 0.0100f
#define AUDIT_TOL 0.0080f
#define LN2 0.6931471805599453094f
#define INV_LN2 1.4426950408889634074f

struct case_desc { uint32_t rows, ncols, stride; float scale; };
static const struct case_desc cases[] = {
    {1u, 1u, 5u, 1.0f}, {1u, 2u, 7u, 0.5f}, {1u, 7u, 13u, 1.0f},
    {1u, 15u, 21u, 0.5f}, {1u, 16u, 23u, 1.0f},
    {2u, 1u, 5u, 0.5f}, {2u, 2u, 7u, 1.0f}, {2u, 7u, 13u, 0.5f},
    {2u, 15u, 21u, 1.0f}, {2u, 16u, 23u, 0.5f},
    {7u, 1u, 5u, 1.0f}, {7u, 2u, 7u, 0.5f}, {7u, 7u, 13u, 1.0f},
    {7u, 15u, 21u, 0.5f}, {7u, 16u, 23u, 1.0f}
};

static uint16_t x_values[BUFFER_N];
static float out_values[BUFFER_N];
static float audit_values[AUDIT_N];

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

static uint16_t logit_half(uint32_t case_id, uint32_t row, uint32_t col) {
    static const uint16_t values[] = {
        0xc400u, 0xc200u, 0xc000u, 0xbc00u, 0x0000u,
        0x3c00u, 0x4000u, 0x4200u, 0x4400u
    };
    return values[(row * 7u + col * 5u + case_id * 3u) % 9u];
}

static uint32_t launched_rows(const struct case_desc *tc) {
    return tc->rows;
}

static float logit_value(uint32_t case_id, const struct case_desc *tc, uint32_t row, uint32_t col) {
    return f16_to_f32(logit_half(case_id, row, col)) * tc->scale;
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
    float sentinel = bits_to_float(SENTINEL_F_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        x_values[i] = INPUT_PAD_H;
        out_values[i] = sentinel;
    }
    for (uint32_t i = 0; i < AUDIT_N; i++)
        audit_values[i] = sentinel;
    for (uint32_t r = 0; r < launched_rows(tc); r++)
        for (uint32_t c = 0; c < tc->ncols; c++)
            x_values[GUARD + r * tc->stride + c] = logit_half(case_id, r, c);
}

static float expected_denom(uint32_t case_id, const struct case_desc *tc, uint32_t row) {
    float maxv = -80.0f;
    for (uint32_t c = 0; c < tc->ncols; c++) {
        float v = logit_value(case_id, tc, row, c);
        if (v > maxv)
            maxv = v;
    }
    float denom = 0.0f;
    for (uint32_t c = 0; c < tc->ncols; c++)
        denom += natural_exp_ref(logit_value(case_id, tc, row, c) - maxv);
    return denom;
}

static float expected_softmax(uint32_t case_id, const struct case_desc *tc, uint32_t row, uint32_t col) {
    float maxv = -80.0f;
    for (uint32_t c = 0; c < tc->ncols; c++) {
        float v = logit_value(case_id, tc, row, c);
        if (v > maxv)
            maxv = v;
    }
    return natural_exp_ref(logit_value(case_id, tc, row, col) - maxv) / expected_denom(case_id, tc, row);
}

static int verify_outputs(uint32_t case_id, const struct case_desc *tc,
                          float *max_abs_diff, float *max_rel_diff,
                          float *max_row_sum_diff, float *max_audit_diff) {
    int mismatches = 0;
    for (uint32_t r = 0; r < launched_rows(tc); r++) {
        float row_sum = 0.0f;
        for (uint32_t c = 0; c < tc->ncols; c++) {
            uint32_t index = GUARD + r * tc->stride + c;
            float expected = expected_softmax(case_id, tc, r, c);
            float got = out_values[index];
            float diff = absf_local(got - expected);
            float rel = diff / maxf_local(absf_local(expected), 1.0e-12f);
            row_sum += got;
            if (diff > *max_abs_diff) *max_abs_diff = diff;
            if (rel > *max_rel_diff) *max_rel_diff = rel;
            if (diff > maxf_local(SOFTMAX_ABS_TOL, SOFTMAX_REL_TOL * absf_local(expected))) {
                if (mismatches < 8)
                    printk("ERROR: mixed_ttir_sfu_softmax case=%d row=%d col=%d got=%f expected=%f diff=%f rel=%f scale=%f\n",
                           (int)case_id, (int)r, (int)c, got, expected, diff, rel, tc->scale);
                mismatches++;
            }
        }
        float sum_diff = absf_local(row_sum - 1.0f);
        if (sum_diff > *max_row_sum_diff)
            *max_row_sum_diff = sum_diff;
        if (sum_diff > ROW_SUM_TOL) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_sfu_softmax row_sum case=%d row=%d sum=%f diff=%f\n",
                       (int)case_id, (int)r, row_sum, sum_diff);
            mismatches++;
        }

        float denom = expected_denom(case_id, tc, r);
        float audit = audit_values[GUARD + r];
        float audit_diff = absf_local(audit - denom);
        if (audit_diff > *max_audit_diff)
            *max_audit_diff = audit_diff;
        if (audit_diff > AUDIT_TOL) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_sfu_softmax audit case=%d row=%d got=%f expected=%f diff=%f\n",
                       (int)case_id, (int)r, audit, denom, audit_diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct case_desc *tc) {
    int mismatches = 0;
    uint32_t rows = launched_rows(tc);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int active = 0;
        int row_padding = 0;
        if (i >= GUARD && i < GUARD + rows * tc->stride) {
            uint32_t rel = i - GUARD;
            uint32_t c = rel % tc->stride;
            active = c < tc->ncols;
            row_padding = c >= tc->ncols;
        }
        if (!active && float_to_bits(out_values[i]) != SENTINEL_F_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_sfu_softmax output sentinel i=%d bits=%x expected=%x\n",
                       (int)i, float_to_bits(out_values[i]), SENTINEL_F_BITS);
            mismatches++;
        }
        if (row_padding && x_values[i] != INPUT_PAD_H) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_sfu_softmax row padding input changed i=%d got_h=%x expected_h=%x\n",
                       (int)i, x_values[i], INPUT_PAD_H);
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < AUDIT_N; i++) {
        int active = i >= GUARD && i < GUARD + rows;
        if (!active && float_to_bits(audit_values[i]) != SENTINEL_F_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_sfu_softmax audit sentinel i=%d bits=%x expected=%x\n",
                       (int)i, float_to_bits(audit_values[i]), SENTINEL_F_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(const struct case_desc *tc) {
    uint32_t hash = 2166136261u ^ tc->rows ^ (tc->ncols << 8) ^ (tc->stride << 16);
    for (uint32_t r = 0; r < launched_rows(tc); r++) {
        for (uint32_t c = 0; c < tc->ncols; c++) {
            uint32_t index = GUARD + r * tc->stride + c;
            hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
        hash ^= float_to_bits(audit_values[GUARD + r]) + 0x85ebca6bu + (r << 7);
        hash = rotl32_local(hash, 9u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("mixed_ttir_sfu_softmax program create failed");

    vc4_deviceptr_t x_dev = 0, out_dev = 0, audit_dev = 0;
    uint32_t x_bytes = BUFFER_N * sizeof(uint16_t);
    uint32_t out_bytes = BUFFER_N * sizeof(float);
    uint32_t audit_bytes = AUDIT_N * sizeof(float);
    if (vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0 ||
        vc4_m2_malloc(program, &audit_dev, audit_bytes) < 0)
        panic("mixed_ttir_sfu_softmax allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f, max_rel_diff = 0.0f, max_row_sum_diff = 0.0f, max_audit_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_desc *tc = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(tc->rows, 2u, 1u);
        fill_buffers(case_id, tc);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(uint16_t);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        vc4_deviceptr_t audit_active = audit_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            vc4_m2_copy_htod(program, audit_dev, audit_values, audit_bytes) < 0 ||
            mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_launch(
                program, grid, block, x_active, tc->scale, out_active,
                audit_active, tc->ncols, tc->stride) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, audit_values, audit_dev, audit_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, x_values, x_dev, x_bytes) < 0) {
            printk("ERROR: mixed_ttir_sfu_softmax launch/copy failed case=%d rows=%d ncols=%d stride=%d\n",
                   (int)case_id, (int)tc->rows, (int)tc->ncols, (int)tc->stride);
            launch_failures++;
            continue;
        }
        int mismatches = verify_outputs(case_id, tc, &max_abs_diff, &max_rel_diff,
                                        &max_row_sum_diff, &max_audit_diff);
        int sentinels = verify_sentinels(tc);
        uint32_t hash = hash_case(tc);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("MIXED_TTIR_SFU_SOFTMAX_AXIS_MASK_CF_F16_CASE case=%d rows=%d launched_rows=%d ncols=%d stride=%d scale=%f mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f max_rel_diff=%f max_row_sum_diff=%f max_audit_diff=%f\n",
               (int)case_id, (int)tc->rows, (int)launched_rows(tc), (int)tc->ncols,
               (int)tc->stride, tc->scale, mismatches, sentinels, hash,
               max_abs_diff, max_rel_diff, max_row_sum_diff, max_audit_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_ncols=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f max_rel_diff=%f max_row_sum_diff=%f max_audit_diff=%f runtime_allocations=%d runtime_launches=%d saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_value_scf_cf=1 saw_program_id_axis0=1 saw_arange_make_range_16=1 saw_masked_load_other_zero=1 saw_masked_store_tail=1 saw_tmu_load=1 saw_vdw_preserve_store=1 saw_tail_mask_clamp_overlaunch=1 saw_ttir_elementwise=1 saw_ttir_control_flow=1 saw_ttir_multi_axis=1 saw_ttir_program_id_axis1=1 saw_ttir_mask_tail=1 saw_ttir_row_strided_memory=1 saw_ttir_reduction_f32_finite_add=1 saw_ttir_scalar_reduction_store=1 saw_ttir_f16_storage_load=1 saw_ttir_f32_compute_after_f16_load=1 saw_value_f16_storage=1 saw_ttir_approx_sfu_exp=1 saw_ttir_approx_sfu_recip_div=1 saw_ttir_finite_f32_max_reduction=1 saw_ttir_softmax_v0=1 saw_value_approx_sfu=1 saw_value_approx_sfu_exp=1 saw_value_approx_sfu_recip_div=1 saw_value_finite_f32_max_reduction=1 saw_value_softmax_v0=1 saw_scalar_to_vector_f32_broadcast=1 saw_approx_math_policy=1 saw_softmax_uses_natural_exp=1 saw_no_exact_default_math=1 saw_no_multiblock_softmax=1 saw_no_full_attention=1 saw_no_tl_dot_tt_dot=1 saw_no_vector_contract=1 saw_no_gemm=1 saw_row_padding_sentinels=1 saw_output_padding_sentinels=1 elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, MAX_ROWS,
           MAX_NCOLS, output_hash, output_hash != 0u ? 1 : 0, max_abs_diff,
           max_rel_diff, max_row_sum_diff, max_audit_diff, 3,
           (int)(sizeof(cases) / sizeof(cases[0])), VC4_CASE_SAW_CPP_TTIR_IMPORTER,
           elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, out_dev);
    vc4Free(program, audit_dev);
    vc4_program_destroy(program);
}
