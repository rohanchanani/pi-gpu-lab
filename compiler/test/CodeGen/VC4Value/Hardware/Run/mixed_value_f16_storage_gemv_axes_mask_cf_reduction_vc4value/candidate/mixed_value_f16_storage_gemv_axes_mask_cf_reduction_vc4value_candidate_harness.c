#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define EXTRA_ROWS 2u
#define MAX_ROWS 7u
#define MAX_COLS 32u
#define MAX_KBLOCKS 2u
#define MAX_LDA 48u
#define ACTIVE_N ((MAX_ROWS + EXTRA_ROWS) * MAX_LDA)
#define BUFFER_N (ACTIVE_N + 2u * GUARD)
#define X_BUFFER_N (MAX_COLS + 2u * GUARD)
#define PARTIAL_OUT_N ((MAX_ROWS + EXTRA_ROWS) * MAX_KBLOCKS + 2u * GUARD)
#define SENTINEL_F_BITS 0xc57d0000u
#define SENTINEL_I 0x6aa3c241
#define SENTINEL_H 0x6c5cu
#define INPUT_PAD_I 0x7331a90d
#define INPUT_PAD_H 0x5c6cu
#define THRESHOLD_I 500
#define THRESHOLD_F 1.75f
#define EPSILON 0.02f

struct case_config {
    uint32_t rows;
    uint32_t cols;
    uint32_t lda;
    uint32_t trip;
    int32_t use_i32_path;
};

static const struct case_config cases[] = {
    {0u, 0u, 20u, 0u, 0},
    {1u, 0u, 20u, 1u, 1},
    {1u, 1u, 20u, 2u, 0},
    {2u, 7u, 20u, 3u, 1},
    {3u, 15u, 22u, 1u, 0},
    {4u, 16u, 24u, 2u, 1},
    {7u, 17u, 41u, 3u, 0},
    {7u, 32u, 48u, 2u, 1},
};

static int32_t rank_i_values[BUFFER_N];
static uint16_t rank_f16_values[BUFFER_N];
static uint16_t x_f16_values[X_BUFFER_N];
static int32_t tail_out_i[BUFFER_N];
static uint16_t tail_out_f16[BUFFER_N];
static float partial_out_f[PARTIAL_OUT_N];
static int32_t expected_tail_i[BUFFER_N];
static float expected_tail_f[BUFFER_N];
static float expected_partial_f[PARTIAL_OUT_N];

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

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    if (exp == 0u) return bits_to_float(sign);
    uint32_t bits = sign | ((exp + 112u) << 23) | (mant << 13);
    return bits_to_float(bits);
}

static uint32_t col_blocks(uint32_t cols) {
    uint32_t blocks = (cols + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static int32_t loop_i_bias(uint32_t trip) {
    return trip > 0u ? 2 : 1;
}

static float loop_f_bias(uint32_t trip) {
    return trip > 0u ? 2.0f : 1.0f;
}

static int32_t rank_i_value(uint32_t r, uint32_t c, const struct case_config *cfg) {
    int32_t value = (int32_t)(120 + r * 97u + c * 23u + cfg->lda);
    if (((r * 5u + c + cfg->trip) & 3u) == 0u)
        value = -value - 17;
    return value;
}

static uint16_t rank_f_half(uint32_t r, uint32_t c, const struct case_config *cfg) {
    static const uint16_t values[] = {
        0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x4000u, 0xc000u, 0x3400u, 0xb400u,
        0x4200u, 0xc200u, 0x3a00u, 0xba00u, 0x4400u, 0xc400u, 0x3000u, 0xb000u
    };
    return values[(r * 7u + c * 3u + cfg->cols + cfg->trip) & 15u];
}

static float rank_f_value(uint32_t r, uint32_t c, const struct case_config *cfg) {
    return f16_to_f32(rank_f_half(r, c, cfg));
}

static uint16_t x_half(uint32_t c, const struct case_config *cfg) {
    static const uint16_t values[] = {
        0x3800u, 0x3a00u, 0xb800u, 0x3400u, 0xba00u, 0x3000u, 0x3c00u, 0xbc00u,
        0x4000u, 0xc000u, 0x3600u, 0xb600u, 0x4200u, 0xc200u, 0x3200u, 0xb200u
    };
    return values[(c * 5u + cfg->rows * 3u + cfg->trip) & 15u];
}

static float x_value(uint32_t c, const struct case_config *cfg) {
    return f16_to_f32(x_half(c, cfg));
}

static void fill_buffers(const struct case_config *cfg) {
    float sentinel_f = bits_to_float(SENTINEL_F_BITS);
    int32_t i_bias = loop_i_bias(cfg->trip);
    float f_bias = loop_f_bias(cfg->trip);
    uint32_t blocks = col_blocks(cfg->cols);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        rank_i_values[i] = INPUT_PAD_I;
        rank_f16_values[i] = INPUT_PAD_H;
        tail_out_i[i] = SENTINEL_I;
        tail_out_f16[i] = SENTINEL_H;
        expected_tail_i[i] = SENTINEL_I;
        expected_tail_f[i] = sentinel_f;
    }
    for (uint32_t i = 0; i < X_BUFFER_N; i++)
        x_f16_values[i] = INPUT_PAD_H;
    for (uint32_t i = 0; i < PARTIAL_OUT_N; i++) {
        partial_out_f[i] = sentinel_f;
        expected_partial_f[i] = sentinel_f;
    }

    for (uint32_t c = 0; c < cfg->cols; c++)
        x_f16_values[GUARD + c] = x_half(c, cfg);

    for (uint32_t r = 0; r < cfg->rows; r++) {
        for (uint32_t kb = 0; kb < blocks; kb++) {
            float partial_dot = 0.0f;
            uint32_t begin = kb * LANES;
            uint32_t end = begin + LANES;
            if (end > cfg->cols)
                end = cfg->cols;
            for (uint32_t c = begin; c < end; c++) {
                uint32_t index = GUARD + r * cfg->lda + c;
                int32_t base_i = rank_i_value(r, c, cfg);
                float base_f = rank_f_value(r, c, cfg);
                float xv = x_value(c, cfg);
                float prod = base_f * xv;
                rank_i_values[index] = base_i;
                rank_f16_values[index] = rank_f_half(r, c, cfg);

                int32_t selected_i;
                float selected_f;
                if (cfg->use_i32_path) {
                    int32_t candidate_i = base_i + i_bias;
                    int cond = candidate_i > THRESHOLD_I;
                    float candidate = base_f + prod + f_bias;
                    selected_i = cond ? candidate_i : base_i;
                    selected_f = cond ? candidate : base_f;
                } else {
                    int cond = base_f < THRESHOLD_F;
                    int32_t candidate_i = base_i + i_bias;
                    float candidate = prod + f_bias;
                    selected_i = cond ? candidate_i : base_i;
                    selected_f = cond ? candidate : base_f;
                }
                expected_tail_i[index] = selected_i;
                expected_tail_f[index] = selected_f;
                partial_dot += prod;
            }
            uint32_t out_index = GUARD + r * blocks + kb;
            expected_partial_f[out_index] = partial_dot;
        }
    }
}

static int verify_tail_results(const struct case_config *cfg, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t r = 0; r < cfg->rows; r++) {
        for (uint32_t c = 0; c < cfg->cols; c++) {
            uint32_t index = GUARD + r * cfg->lda + c;
            float got_f = f16_to_f32(tail_out_f16[index]);
            float diff = got_f - expected_tail_f[index];
            float ad = absf_local(diff);
            if (ad > *max_abs_diff)
                *max_abs_diff = ad;
            if (ad > EPSILON || tail_out_i[index] != expected_tail_i[index]) {
                if (mismatches < 8)
                    printk("ERROR: mixed f16 gemv tail row=%d col=%d got_f=%f exp_f=%f got_h=%x got_i=%d exp_i=%d diff=%f\n",
                           (int)r, (int)c, got_f, expected_tail_f[index],
                           tail_out_f16[index], tail_out_i[index], expected_tail_i[index], diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_partials(const struct case_config *cfg, float *max_abs_diff) {
    int mismatches = 0;
    uint32_t blocks = col_blocks(cfg->cols);
    for (uint32_t r = 0; r < cfg->rows; r++) {
        for (uint32_t kb = 0; kb < blocks; kb++) {
            uint32_t index = GUARD + r * blocks + kb;
            float diff = partial_out_f[index] - expected_partial_f[index];
            float ad = absf_local(diff);
            if (ad > *max_abs_diff)
                *max_abs_diff = ad;
            if (ad > EPSILON) {
                if (mismatches < 8)
                    printk("ERROR: mixed f16 gemv partial row=%d kblock=%d got=%f exp=%f diff=%f\n",
                           (int)r, (int)kb, partial_out_f[index],
                           expected_partial_f[index], diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct case_config *cfg) {
    int mismatches = 0;
    uint32_t blocks = col_blocks(cfg->cols);
    uint32_t active_partials = cfg->rows * blocks;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int active = 0;
        int row_padding = 0;
        if (i >= GUARD && i < GUARD + cfg->rows * cfg->lda) {
            uint32_t rel = i - GUARD;
            uint32_t c = rel % cfg->lda;
            active = c < cfg->cols;
            row_padding = c >= cfg->cols;
        }
        if (!active && (tail_out_i[i] != SENTINEL_I || tail_out_f16[i] != SENTINEL_H)) {
            if (mismatches < 8)
                printk("ERROR: mixed f16 gemv tail sentinel i=%d got_i=%x got_h=%x\n",
                       (int)i, (uint32_t)tail_out_i[i], tail_out_f16[i]);
            mismatches++;
        }
        if (row_padding && (rank_i_values[i] != INPUT_PAD_I || rank_f16_values[i] != INPUT_PAD_H)) {
            if (mismatches < 8)
                printk("ERROR: mixed f16 gemv row padding input changed i=%d got_i=%x got_h=%x\n",
                       (int)i, (uint32_t)rank_i_values[i], rank_f16_values[i]);
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < PARTIAL_OUT_N; i++) {
        int active_partial = i >= GUARD && i < GUARD + active_partials;
        if (!active_partial && float_to_bits(partial_out_f[i]) != SENTINEL_F_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed f16 gemv partial sentinel i=%d got=%x\n",
                       (int)i, float_to_bits(partial_out_f[i]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(const struct case_config *cfg) {
    uint32_t blocks = col_blocks(cfg->cols);
    uint32_t hash = 2166136261u ^ cfg->rows ^ (cfg->cols << 8) ^ (cfg->lda << 16);
    for (uint32_t r = 0; r < cfg->rows; r++) {
        for (uint32_t c = 0; c < cfg->cols; c++) {
            uint32_t index = GUARD + r * cfg->lda + c;
            hash ^= (uint32_t)tail_out_i[index] + tail_out_f16[index] +
                    0x85ebca6bu + (index << 5);
            hash = rotl32_local(hash, 7u) * 16777619u;
        }
        for (uint32_t kb = 0; kb < blocks; kb++) {
            uint32_t partial_index = GUARD + r * blocks + kb;
            hash ^= float_to_bits(partial_out_f[partial_index]) + 0xc2b2ae35u + (kb << 4);
            hash = rotl32_local(hash, 11u) * 16777619u;
        }
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("mixed_value_f16_gemv program create failed");

    uint32_t bytes_i = BUFFER_N * sizeof(int32_t);
    uint32_t bytes_h = BUFFER_N * sizeof(uint16_t);
    uint32_t x_bytes_h = X_BUFFER_N * sizeof(uint16_t);
    uint32_t partial_bytes_f = PARTIAL_OUT_N * sizeof(float);
    vc4_deviceptr_t rank_i_dev = 0, rank_f16_dev = 0, x_f16_dev = 0;
    vc4_deviceptr_t partial_f_dev = 0;
    vc4_deviceptr_t tail_i_dev = 0, tail_f16_dev = 0;
    if (vc4_m2_malloc(program, &rank_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &rank_f16_dev, bytes_h) < 0 ||
        vc4_m2_malloc(program, &x_f16_dev, x_bytes_h) < 0 ||
        vc4_m2_malloc(program, &partial_f_dev, partial_bytes_f) < 0 ||
        vc4_m2_malloc(program, &tail_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &tail_f16_dev, bytes_h) < 0)
        panic("mixed_value_f16_gemv allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    float max_abs_diff_overall = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_config *cfg = &cases[case_id];
        uint32_t blocks = col_blocks(cfg->cols);
        uint32_t partial_count = cfg->rows * blocks;
        vc4_dim3 grid = vc4_m2_dim3(blocks, cfg->rows + EXTRA_ROWS, 1u);
        fill_buffers(cfg);
        vc4_deviceptr_t rank_i_active = rank_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t rank_f16_active = rank_f16_dev + GUARD * sizeof(uint16_t);
        vc4_deviceptr_t x_f16_active = x_f16_dev + GUARD * sizeof(uint16_t);
        vc4_deviceptr_t partial_f_active = partial_f_dev + GUARD * sizeof(float);
        vc4_deviceptr_t tail_i_active = tail_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t tail_f16_active = tail_f16_dev + GUARD * sizeof(uint16_t);
        if (vc4_m2_copy_htod(program, rank_i_dev, rank_i_values, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, rank_f16_dev, rank_f16_values, bytes_h) < 0 ||
            vc4_m2_copy_htod(program, x_f16_dev, x_f16_values, x_bytes_h) < 0 ||
            vc4_m2_copy_htod(program, partial_f_dev, partial_out_f, partial_bytes_f) < 0 ||
            vc4_m2_copy_htod(program, tail_i_dev, tail_out_i, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, tail_f16_dev, tail_out_f16, bytes_h) < 0 ||
            mixed_value_f16_storage_gemv_axes_mask_cf_reduction_vc4value_launch(
                program, grid, block, rank_i_active, rank_f16_active, x_f16_active,
                partial_f_active, tail_i_active, tail_f16_active,
                cfg->rows, cfg->cols, cfg->lda,
                blocks, partial_count, THRESHOLD_I, THRESHOLD_F, cfg->trip,
                cfg->use_i32_path) < 0) {
            printk("ERROR: mixed_value_f16_gemv launch/copy failed case=%d rows=%d cols=%d lda=%d\n",
                   (int)case_id, (int)cfg->rows, (int)cfg->cols, (int)cfg->lda);
            launch_failures++;
            continue;
        }
        if (vc4_m2_copy_dtoh(program, tail_out_i, tail_i_dev, bytes_i) < 0 ||
            vc4_m2_copy_dtoh(program, tail_out_f16, tail_f16_dev, bytes_h) < 0 ||
            vc4_m2_copy_dtoh(program, partial_out_f, partial_f_dev, partial_bytes_f) < 0 ||
            vc4_m2_copy_dtoh(program, rank_i_values, rank_i_dev, bytes_i) < 0 ||
            vc4_m2_copy_dtoh(program, rank_f16_values, rank_f16_dev, bytes_h) < 0) {
            printk("ERROR: mixed_value_f16_gemv dtoh failed case=%d rows=%d cols=%d lda=%d\n",
                   (int)case_id, (int)cfg->rows, (int)cfg->cols, (int)cfg->lda);
            launch_failures++;
            continue;
        }

        float max_abs_diff = 0.0f;
        int tail_mismatches = verify_tail_results(cfg, &max_abs_diff);
        int partial_mismatches = verify_partials(cfg, &max_abs_diff);
        int sentinels = verify_sentinels(cfg);
        uint32_t case_hash = hash_output(cfg);
        total_mismatches += tail_mismatches + partial_mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)(cfg->rows * cfg->cols);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("MIXED_VALUE_F16_STORAGE_GEMV_AXIS_MASK_CF_CASE case=%d grid=%dx%d rows=%d cols=%d lda=%d trip=%d flag=%d tail_mismatches=%d partial_mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)blocks, (int)(cfg->rows + EXTRA_ROWS),
               (int)cfg->rows, (int)cfg->cols, (int)cfg->lda, (int)cfg->trip,
               (int)cfg->use_i32_path, tail_mismatches, partial_mismatches,
               sentinels, case_hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_value_f16_storage_gemv_axes_mask_cf_reduction_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_cols=%d max_lda=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d saw_value_multi_axis=1 saw_value_control_flow=1 saw_value_mask_tail=1 saw_value_compute_mask_select=1 saw_value_row_strided_memory=1 saw_value_reduction_f32_finite_add=1 saw_value_scalar_reduction_store=1 saw_value_gemv_f32_row_dot=1 saw_value_gemv_partial_kblock=1 saw_value_f16_storage_load=1 saw_value_f16_storage_store=1 saw_value_f32_compute_after_f16_load=1 saw_f16_storage_finite_policy=1 saw_no_native_f16_arithmetic=1 saw_no_bf16_fp8=1 saw_no_softmax_sfu=1 saw_no_tl_dot_tt_dot=1 saw_no_vector_contract=1 saw_no_multiblock_k_accumulation=1 elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_ROWS, MAX_COLS, MAX_LDA, output_hash,
           output_hash != 0u ? 1 : 0, max_abs_diff_overall, 6,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, rank_i_dev);
    vc4Free(program, rank_f16_dev);
    vc4Free(program, x_f16_dev);
    vc4Free(program, partial_f_dev);
    vc4Free(program, tail_i_dev);
    vc4Free(program, tail_f16_dev);
    vc4_program_destroy(program);
}
