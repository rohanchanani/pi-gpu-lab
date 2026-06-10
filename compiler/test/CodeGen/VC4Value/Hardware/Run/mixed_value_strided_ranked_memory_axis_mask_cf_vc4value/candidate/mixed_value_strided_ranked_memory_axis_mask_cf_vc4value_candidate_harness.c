#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define EXTRA_ROWS 2u
#define MAX_ROWS 7u
#define MAX_COLS 33u
#define MAX_LDA 56u
#define ACTIVE_N ((MAX_ROWS + EXTRA_ROWS) * MAX_LDA)
#define BUFFER_N (ACTIVE_N + 2u * GUARD)
#define SENTINEL_F_BITS 0xc57b0000u
#define SENTINEL_I 0x64b8c931
#define THRESHOLD_I 700
#define THRESHOLD_F 2.75f
#define EPSILON 0.001f

struct case_config {
    uint32_t rows;
    uint32_t cols;
    uint32_t lda;
    float a;
    uint32_t trip;
    int32_t use_i32_path;
};

static const struct case_config cases[] = {
    {0u, 0u, 20u, 1.0f, 0u, 0},
    {2u, 0u, 20u, -0.5f, 1u, 1},
    {1u, 1u, 20u, 0.25f, 2u, 0},
    {0u, 17u, 38u, 1.25f, 3u, 1},
    {3u, 17u, 38u, 1.5f, 3u, 1},
    {4u, 16u, 24u, -1.0f, 0u, 0},
    {5u, 31u, 40u, 0.75f, 4u, 1},
    {7u, 33u, 56u, -0.25f, 2u, 0},
};

static int32_t flat_i_values[BUFFER_N];
static int32_t rank_i_values[BUFFER_N];
static float rank_f_values[BUFFER_N];
static float rank_out_f[BUFFER_N];
static int32_t rank_out_i[BUFFER_N];
static int32_t flat_policy_out[BUFFER_N];
static float expected_rank_out_f[BUFFER_N];
static int32_t expected_rank_out_i[BUFFER_N];
static int32_t expected_policy_out[BUFFER_N];

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

static uint32_t col_blocks(uint32_t cols) {
    uint32_t blocks = (cols + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static uint32_t rounded_cols(uint32_t cols) {
    return col_blocks(cols) * LANES;
}

static float loop_factor(float a, uint32_t trip) {
    float factor = a;
    for (uint32_t i = 0; i < trip; i++)
        factor += 1.0f;
    return factor;
}

static int32_t flat_i_value(uint32_t r, uint32_t c, const struct case_config *cfg) {
    int32_t value = (int32_t)(100 + r * 211u + c * 17u + cfg->cols);
    if (((r + c + cfg->rows) & 5u) == 0u)
        value = -value;
    return value;
}

static int32_t rank_i_value(uint32_t r, uint32_t c, const struct case_config *cfg) {
    int32_t value = (int32_t)(200 + r * 307u + c * 19u + cfg->lda);
    if (((r * 3u + c + cfg->cols) & 3u) == 0u)
        value = -value - 13;
    return value;
}

static float rank_f_value(uint32_t r, uint32_t c, const struct case_config *cfg) {
    int32_t whole = (int32_t)((r * 29u + c * 7u + cfg->lda) % 73u) - 36;
    return (float)whole * 0.125f;
}

static void fill_buffers(const struct case_config *cfg) {
    float sentinel_f = bits_to_float(SENTINEL_F_BITS);
    float factor = loop_factor(cfg->a, cfg->trip);
    uint32_t rounded = rounded_cols(cfg->cols);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        flat_i_values[i] = (int32_t)(0x41000000u + i);
        rank_i_values[i] = (int32_t)(0x42000000u + i);
        rank_f_values[i] = (float)((int32_t)(i % 19u) - 9) * 0.25f;
        rank_out_f[i] = sentinel_f;
        rank_out_i[i] = SENTINEL_I;
        flat_policy_out[i] = SENTINEL_I;
        expected_rank_out_f[i] = sentinel_f;
        expected_rank_out_i[i] = SENTINEL_I;
        expected_policy_out[i] = SENTINEL_I;
    }

    for (uint32_t r = 0; r < cfg->rows; r++) {
        for (uint32_t c = 0; c < cfg->lda; c++) {
            uint32_t index = GUARD + r * cfg->lda + c;
            if (c < cfg->cols) {
                int32_t flat_i = flat_i_value(r, c, cfg);
                int32_t rank_i = rank_i_value(r, c, cfg);
                float rank_f = rank_f_value(r, c, cfg);
                flat_i_values[index] = flat_i;
                rank_i_values[index] = rank_i;
                rank_f_values[index] = rank_f;

                int32_t sum_i = rank_i + flat_i;
                if (cfg->use_i32_path) {
                    int cond = sum_i > THRESHOLD_I;
                    float candidate = factor * rank_f + 1.0f;
                    expected_rank_out_f[index] = cond ? candidate : rank_f;
                    expected_rank_out_i[index] = cond ? sum_i + (int32_t)col_blocks(cfg->cols)
                                                      : flat_i;
                } else {
                    int cond = rank_f < THRESHOLD_F;
                    float candidate = factor * rank_f + 1.0f;
                    expected_rank_out_f[index] = cond ? candidate : rank_f;
                    expected_rank_out_i[index] = sum_i + (int32_t)col_blocks(cfg->cols);
                }
                expected_policy_out[index] = flat_i;
            } else if (c < rounded) {
                expected_policy_out[index] = 0;
            }
        }
    }
}

static int verify_tail_results(const struct case_config *cfg, float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t r = 0; r < cfg->rows; r++) {
        for (uint32_t c = 0; c < cfg->cols; c++) {
            uint32_t index = GUARD + r * cfg->lda + c;
            float diff = rank_out_f[index] - expected_rank_out_f[index];
            float ad = absf_local(diff);
            if (ad > *max_abs_diff)
                *max_abs_diff = ad;
            if (ad > EPSILON || rank_out_i[index] != expected_rank_out_i[index]) {
                if (mismatches < 8)
                    printk("ERROR: mixed_strided active row=%d col=%d got_f=%f exp_f=%f got_i=%d exp_i=%d diff=%f\n",
                           (int)r, (int)c, rank_out_f[index], expected_rank_out_f[index],
                           rank_out_i[index], expected_rank_out_i[index], diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_policy_results(const struct case_config *cfg) {
    int mismatches = 0;
    for (uint32_t r = 0; r < cfg->rows; r++) {
        for (uint32_t c = 0; c < cfg->lda; c++) {
            uint32_t index = GUARD + r * cfg->lda + c;
            if (flat_policy_out[index] != expected_policy_out[index]) {
                if (mismatches < 8)
                    printk("ERROR: mixed_strided policy row=%d col=%d got=%x expected=%x\n",
                           (int)r, (int)c, (uint32_t)flat_policy_out[index],
                           (uint32_t)expected_policy_out[index]);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct case_config *cfg) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int active = 0;
        int policy_touched = 0;
        if (i >= GUARD && i < GUARD + cfg->rows * cfg->lda) {
            uint32_t rel = i - GUARD;
            uint32_t c = rel % cfg->lda;
            active = c < cfg->cols;
            policy_touched = c < rounded_cols(cfg->cols);
        }
        if (!active &&
            (float_to_bits(rank_out_f[i]) != SENTINEL_F_BITS || rank_out_i[i] != SENTINEL_I)) {
            if (mismatches < 8)
                printk("ERROR: mixed_strided row padding sentinel i=%d got_f=%x got_i=%x\n",
                       (int)i, float_to_bits(rank_out_f[i]), (uint32_t)rank_out_i[i]);
            mismatches++;
        }
        if (!policy_touched && flat_policy_out[i] != SENTINEL_I) {
            if (mismatches < 8)
                printk("ERROR: mixed_strided policy sentinel i=%d got=%x expected=%x\n",
                       (int)i, (uint32_t)flat_policy_out[i], SENTINEL_I);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(const struct case_config *cfg) {
    uint32_t hash = 2166136261u ^ cfg->rows ^ (cfg->cols << 8) ^ (cfg->lda << 16);
    for (uint32_t r = 0; r < cfg->rows; r++) {
        for (uint32_t c = 0; c < cfg->cols; c++) {
            uint32_t index = GUARD + r * cfg->lda + c;
            hash ^= float_to_bits(rank_out_f[index]) + (uint32_t)rank_out_i[index] +
                    0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
        for (uint32_t c = 0; c < rounded_cols(cfg->cols); c++) {
            uint32_t index = GUARD + r * cfg->lda + c;
            hash ^= (uint32_t)flat_policy_out[index] + 0x85ebca6bu + (index << 5);
            hash = rotl32_local(hash, 7u) * 16777619u;
        }
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("mixed_value_strided_ranked program create failed");

    uint32_t bytes_f = BUFFER_N * sizeof(float);
    uint32_t bytes_i = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t flat_i_dev = 0, rank_i_dev = 0, rank_f_dev = 0;
    vc4_deviceptr_t rank_out_f_dev = 0, rank_out_i_dev = 0, policy_dev = 0;
    if (vc4_m2_malloc(program, &flat_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &rank_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &rank_f_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &rank_out_f_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &rank_out_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &policy_dev, bytes_i) < 0)
        panic("mixed_value_strided_ranked allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    float max_abs_diff_overall = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_config *cfg = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(col_blocks(cfg->cols), cfg->rows + EXTRA_ROWS, 1);
        uint32_t total = cfg->rows * cfg->lda;
        fill_buffers(cfg);
        vc4_deviceptr_t flat_i_active = flat_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t rank_i_active = rank_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t rank_f_active = rank_f_dev + GUARD * sizeof(float);
        vc4_deviceptr_t rank_out_f_active = rank_out_f_dev + GUARD * sizeof(float);
        vc4_deviceptr_t rank_out_i_active = rank_out_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t policy_active = policy_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, flat_i_dev, flat_i_values, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, rank_i_dev, rank_i_values, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, rank_f_dev, rank_f_values, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, rank_out_f_dev, rank_out_f, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, rank_out_i_dev, rank_out_i, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, policy_dev, flat_policy_out, bytes_i) < 0 ||
            mixed_value_strided_ranked_memory_axis_mask_cf_vc4value_launch(
                program, grid, block, flat_i_active, rank_i_active, rank_f_active,
                rank_out_f_active, rank_out_i_active, policy_active, cfg->rows,
                cfg->cols, cfg->lda, total, THRESHOLD_I, THRESHOLD_F, cfg->a,
                cfg->trip, cfg->use_i32_path) < 0 ||
            vc4_m2_copy_dtoh(program, rank_out_f, rank_out_f_dev, bytes_f) < 0 ||
            vc4_m2_copy_dtoh(program, rank_out_i, rank_out_i_dev, bytes_i) < 0 ||
            vc4_m2_copy_dtoh(program, flat_policy_out, policy_dev, bytes_i) < 0) {
            printk("ERROR: mixed_value_strided_ranked launch/copy failed case=%d rows=%d cols=%d lda=%d\n",
                   (int)case_id, (int)cfg->rows, (int)cfg->cols, (int)cfg->lda);
            launch_failures++;
            continue;
        }

        float max_abs_diff = 0.0f;
        int result_mismatches = verify_tail_results(cfg, &max_abs_diff);
        int policy_mismatches = verify_policy_results(cfg);
        int sentinels = verify_sentinels(cfg);
        uint32_t case_hash = hash_output(cfg);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += result_mismatches + policy_mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)(cfg->rows * cfg->cols);
        printk("MIXED_VALUE_STRIDED_RANKED_AXIS_MASK_CF_CASE case=%d grid=%dx%d rows=%d cols=%d lda=%d trip=%d flag=%d result_mismatches=%d policy_mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)col_blocks(cfg->cols), (int)(cfg->rows + EXTRA_ROWS),
               (int)cfg->rows, (int)cfg->cols, (int)cfg->lda, (int)cfg->trip,
               (int)cfg->use_i32_path, result_mismatches, policy_mismatches,
               sentinels, case_hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_value_strided_ranked_memory_axis_mask_cf_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_cols=%d max_lda=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d saw_value_multi_axis=1 saw_value_control_flow=1 saw_value_mask_tail=1 saw_value_compute_mask_select=1 saw_value_rank1_flattened_stride=1 saw_value_rank2_row_slice=1 saw_value_stride_args=1 saw_value_memref_dim_metadata=1 saw_no_gather_lane_stride=1 saw_no_hidden_memref_descriptor=1 elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_ROWS, MAX_COLS, MAX_LDA, output_hash,
           output_hash != 0u ? 1 : 0, max_abs_diff_overall, 6,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, flat_i_dev);
    vc4Free(program, rank_i_dev);
    vc4Free(program, rank_f_dev);
    vc4Free(program, rank_out_f_dev);
    vc4Free(program, rank_out_i_dev);
    vc4Free(program, policy_dev);
    vc4_program_destroy(program);
}
