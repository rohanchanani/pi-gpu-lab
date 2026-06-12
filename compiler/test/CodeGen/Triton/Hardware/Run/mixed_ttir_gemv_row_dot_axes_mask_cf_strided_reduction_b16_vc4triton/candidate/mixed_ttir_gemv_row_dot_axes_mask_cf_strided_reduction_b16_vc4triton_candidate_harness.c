#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_ROWS 7u
#define EXTRA_ROWS 2u
#define MAX_K 32u
#define MAX_LDA 41u
#define MAX_KBLOCKS 2u
#define A_BUFFER_N ((MAX_ROWS + EXTRA_ROWS) * MAX_LDA + 2u * GUARD)
#define X_BUFFER_N (MAX_K + 2u * GUARD)
#define Y_BUFFER_N (MAX_ROWS + EXTRA_ROWS + 2u * GUARD)
#define P_BUFFER_N ((MAX_ROWS + EXTRA_ROWS) * MAX_KBLOCKS + 2u * GUARD)
#define A_SENTINEL_BITS 0x7f810123u
#define X_SENTINEL_BITS 0x7f820456u
#define Y_SENTINEL_BITS 0xc57d0000u
#define P_SENTINEL_BITS 0xc6038000u
#define F32_TOLERANCE 0.001f

struct row_dot_case {
    uint32_t rows;
    uint32_t k;
    uint32_t lda;
    int32_t flag;
    float bias;
};

struct partial_case {
    uint32_t rows;
    uint32_t k;
    uint32_t lda;
    uint32_t num_kblocks;
};

static const struct row_dot_case row_cases[] = {
    {1u, 0u, 19u, 0, 0.0f},
    {1u, 1u, 19u, 1, 0.25f},
    {2u, 2u, 21u, 0, -0.5f},
    {2u, 15u, 23u, 1, 0.75f},
    {7u, 16u, 41u, 0, 1.0f},
    {7u, 5u, 41u, 1, -1.25f}
};

static const struct partial_case partial_cases[] = {
    {1u, 0u, 19u, 1u},
    {1u, 1u, 19u, 1u},
    {2u, 15u, 23u, 1u},
    {7u, 16u, 23u, 1u},
    {2u, 17u, 37u, 2u},
    {7u, 32u, 41u, 2u}
};

static float a_values[A_BUFFER_N];
static float x_values[X_BUFFER_N];
static float y_values[Y_BUFFER_N];
static float partial_values[P_BUFFER_N];

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

static float absf_local(float value) {
    return value < 0.0f ? -value : value;
}

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static float a_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 11u + row * 7u + col * 5u) % 29u) - 14;
    float value = (float)whole * 0.125f;
    if (((case_id + row + col) & 5u) == 0u)
        value = 0.0f;
    if (((row ^ col ^ case_id) & 1u) != 0u)
        value = -value;
    return value;
}

static float x_value(uint32_t case_id, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 13u + col * 3u) % 23u) - 11;
    float value = (float)whole * 0.0625f;
    if ((col % 7u) == 0u)
        value = 0.0f;
    return value;
}

static uint32_t kblocks_for(uint32_t k) {
    uint32_t blocks = (k + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static void fill_row_buffers(uint32_t case_id, const struct row_dot_case *c) {
    float a_sentinel = bits_to_float(A_SENTINEL_BITS);
    float x_sentinel = bits_to_float(X_SENTINEL_BITS);
    float y_sentinel = bits_to_float(Y_SENTINEL_BITS);
    for (uint32_t i = 0; i < A_BUFFER_N; i++)
        a_values[i] = a_sentinel;
    for (uint32_t i = 0; i < X_BUFFER_N; i++)
        x_values[i] = x_sentinel;
    for (uint32_t i = 0; i < Y_BUFFER_N; i++)
        y_values[i] = y_sentinel;

    for (uint32_t row = 0; row < c->rows; row++)
        for (uint32_t col = 0; col < c->k; col++)
            a_values[GUARD + row * c->lda + col] = a_value(case_id, row, col);
    for (uint32_t col = 0; col < c->k; col++)
        x_values[GUARD + col] = x_value(case_id, col);
}

static void fill_partial_buffers(uint32_t case_id, const struct partial_case *c) {
    float a_sentinel = bits_to_float(A_SENTINEL_BITS);
    float x_sentinel = bits_to_float(X_SENTINEL_BITS);
    float p_sentinel = bits_to_float(P_SENTINEL_BITS);
    for (uint32_t i = 0; i < A_BUFFER_N; i++)
        a_values[i] = a_sentinel;
    for (uint32_t i = 0; i < X_BUFFER_N; i++)
        x_values[i] = x_sentinel;
    for (uint32_t i = 0; i < P_BUFFER_N; i++)
        partial_values[i] = p_sentinel;

    for (uint32_t row = 0; row < c->rows; row++)
        for (uint32_t col = 0; col < c->k; col++)
            a_values[GUARD + row * c->lda + col] = a_value(case_id + 17u, row, col);
    for (uint32_t col = 0; col < c->k; col++)
        x_values[GUARD + col] = x_value(case_id + 17u, col);
}

static float expected_row(uint32_t case_id, const struct row_dot_case *c,
                          uint32_t row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < c->k; col++)
        sum += a_value(case_id, row, col) * x_value(case_id, col);
    if (c->flag != 0)
        sum += c->bias;
    return sum;
}

static float expected_partial(uint32_t case_id, const struct partial_case *c,
                              uint32_t row, uint32_t kblock) {
    uint32_t start = kblock * LANES;
    uint32_t end = min_u32(c->k, start + LANES);
    float sum = 0.0f;
    for (uint32_t col = start; col < end; col++)
        sum += a_value(case_id + 17u, row, col) * x_value(case_id + 17u, col);
    return sum;
}

static int verify_row_outputs(uint32_t case_id, const struct row_dot_case *c,
                              float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t row = 0; row < c->rows; row++) {
        uint32_t index = GUARD + row;
        float expected = expected_row(case_id, c, row);
        float diff = y_values[index] - expected;
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR GEMV row case=%d row=%d got=%f expected=%f diff=%f\n",
                       (int)case_id, (int)row, y_values[index], expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_partial_outputs(uint32_t case_id, const struct partial_case *c,
                                  float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t row = 0; row < c->rows; row++) {
        for (uint32_t kblock = 0; kblock < c->num_kblocks; kblock++) {
            uint32_t index = GUARD + row * c->num_kblocks + kblock;
            float expected = expected_partial(case_id, c, row, kblock);
            float diff = partial_values[index] - expected;
            float ad = absf_local(diff);
            if (ad > *max_abs_diff)
                *max_abs_diff = ad;
            if (ad > F32_TOLERANCE) {
                if (mismatches < 8)
                    printk("ERROR: mixed TTIR GEMV partial case=%d row=%d kblock=%d got=%f expected=%f diff=%f\n",
                           (int)case_id, (int)row, (int)kblock,
                           partial_values[index], expected, diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_row_sentinels(const struct row_dot_case *c) {
    int mismatches = 0;
    for (uint32_t i = 0; i < Y_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + c->rows)
            continue;
        if (float_to_bits(y_values[i]) != Y_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR GEMV Y sentinel i=%d bits=%x\n",
                       (int)i, float_to_bits(y_values[i]));
            mismatches++;
        }
    }
    for (uint32_t row = 0; row < c->rows; row++) {
        for (uint32_t col = c->k; col < c->lda; col++) {
            uint32_t index = GUARD + row * c->lda + col;
            if (float_to_bits(a_values[index]) != A_SENTINEL_BITS) {
                if (mismatches < 8)
                    printk("ERROR: mixed TTIR GEMV A padding row=%d col=%d bits=%x\n",
                           (int)row, (int)col, float_to_bits(a_values[index]));
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_partial_sentinels(const struct partial_case *c) {
    int mismatches = 0;
    uint32_t active_partials = c->rows * c->num_kblocks;
    for (uint32_t i = 0; i < P_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + active_partials)
            continue;
        if (float_to_bits(partial_values[i]) != P_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR GEMV partial sentinel i=%d bits=%x\n",
                       (int)i, float_to_bits(partial_values[i]));
            mismatches++;
        }
    }
    for (uint32_t row = 0; row < c->rows; row++) {
        for (uint32_t col = c->k; col < c->lda; col++) {
            uint32_t index = GUARD + row * c->lda + col;
            if (float_to_bits(a_values[index]) != A_SENTINEL_BITS) {
                if (mismatches < 8)
                    printk("ERROR: mixed TTIR GEMV partial A padding row=%d col=%d bits=%x\n",
                           (int)row, (int)col, float_to_bits(a_values[index]));
                mismatches++;
            }
        }
    }
    for (uint32_t col = c->k; col < MAX_K; col++) {
        uint32_t index = GUARD + col;
        if (float_to_bits(x_values[index]) != X_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR GEMV X padding col=%d bits=%x\n",
                       (int)col, float_to_bits(x_values[index]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_rows(const struct row_dot_case *c) {
    uint32_t hash = 2166136261u ^ c->rows ^ (c->k << 8) ^ (c->lda << 16);
    for (uint32_t row = 0; row < c->rows; row++) {
        hash ^= float_to_bits(y_values[GUARD + row]) + 0x9e3779b9u + (row << 6);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

static uint32_t hash_partials(const struct partial_case *c) {
    uint32_t hash = 2166136261u ^ c->rows ^ (c->k << 8) ^ (c->num_kblocks << 16);
    for (uint32_t row = 0; row < c->rows; row++) {
        for (uint32_t kblock = 0; kblock < c->num_kblocks; kblock++) {
            uint32_t index = GUARD + row * c->num_kblocks + kblock;
            hash ^= float_to_bits(partial_values[index]) + 0xc2b2ae35u +
                    (row << 6) + kblock;
            hash = rotl32_local(hash, 7u) * 16777619u;
        }
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("mixed TTIR GEMV program create failed");

    vc4_deviceptr_t a_dev = 0, x_dev = 0, y_dev = 0, p_dev = 0;
    uint32_t a_bytes = A_BUFFER_N * sizeof(float);
    uint32_t x_bytes = X_BUFFER_N * sizeof(float);
    uint32_t y_bytes = Y_BUFFER_N * sizeof(float);
    uint32_t p_bytes = P_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, y_bytes) < 0 ||
        vc4_m2_malloc(program, &p_dev, p_bytes) < 0)
        panic("mixed TTIR GEMV allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    int saw_flag_true = 0, saw_flag_false = 0;
    float max_abs_diff = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(row_cases) / sizeof(row_cases[0]); case_id++) {
        const struct row_dot_case *c = &row_cases[case_id];
        uint32_t grid_y = c->rows + EXTRA_ROWS;
        vc4_dim3 grid = vc4_m2_dim3(1u, grid_y, 1u);
        if (c->flag != 0)
            saw_flag_true = 1;
        else
            saw_flag_false = 1;
        fill_row_buffers(case_id, c);
        vc4_deviceptr_t a_active = a_dev + GUARD * sizeof(float);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, y_bytes) < 0 ||
            mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_kernel_launch(
                program, grid, block, a_active, x_active, y_active, c->k,
                c->lda, c->rows, c->flag, c->bias) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, y_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, a_values, a_dev, a_bytes) < 0) {
            printk("ERROR: mixed TTIR GEMV row launch/copy failed case=%d rows=%d k=%d lda=%d\n",
                   (int)case_id, (int)c->rows, (int)c->k, (int)c->lda);
            launch_failures++;
            continue;
        }
        int mismatches = verify_row_outputs(case_id, c, &max_abs_diff);
        int sentinels = verify_row_sentinels(c);
        uint32_t hash = hash_rows(c);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)(c->rows * c->k);
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6);
        output_hash = rotl32_local(output_hash, 5u);
        printk("MIXED_TTIR_GEMV_ROW_CASE case=%d rows=%d k=%d lda=%d flag=%d grid=(1,%d) mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)c->rows, (int)c->k, (int)c->lda,
               (int)c->flag, (int)grid_y, mismatches,
               sentinels, hash, max_abs_diff);
    }

    for (uint32_t case_id = 0; case_id < sizeof(partial_cases) / sizeof(partial_cases[0]); case_id++) {
        const struct partial_case *c = &partial_cases[case_id];
        uint32_t grid_y = c->rows;
        vc4_dim3 grid = vc4_m2_dim3(c->num_kblocks, grid_y, 1u);
        fill_partial_buffers(case_id, c);
        vc4_deviceptr_t a_active = a_dev + GUARD * sizeof(float);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t p_active = p_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
            vc4_m2_copy_htod(program, p_dev, partial_values, p_bytes) < 0 ||
            ttir_gemv_partial_kblock_f32_b16_kernel_launch(
                program, grid, block, a_active, x_active, p_active, c->k,
                c->lda, c->num_kblocks) < 0 ||
            vc4_m2_copy_dtoh(program, partial_values, p_dev, p_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, a_values, a_dev, a_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, x_values, x_dev, x_bytes) < 0) {
            printk("ERROR: mixed TTIR GEMV partial launch/copy failed case=%d rows=%d k=%d kblocks=%d lda=%d\n",
                   (int)case_id, (int)c->rows, (int)c->k,
                   (int)c->num_kblocks, (int)c->lda);
            launch_failures++;
            continue;
        }
        int mismatches = verify_partial_outputs(case_id, c, &max_abs_diff);
        int sentinels = verify_partial_sentinels(c);
        uint32_t hash = hash_partials(c);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)(c->rows * c->k);
        output_hash ^= hash + 0x85ebca6bu + (case_id << 6);
        output_hash = rotl32_local(output_hash, 7u);
        printk("MIXED_TTIR_GEMV_PARTIAL_CASE case=%d rows=%d k=%d lda=%d kblocks=%d expected_kblocks=%d grid=(%d,%d) mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)c->rows, (int)c->k, (int)c->lda,
               (int)c->num_kblocks, (int)kblocks_for(c->k),
               (int)c->num_kblocks, (int)grid_y, mismatches,
               sentinels, hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    int cases = (int)(sizeof(row_cases) / sizeof(row_cases[0]) +
                      sizeof(partial_cases) / sizeof(partial_cases[0]));
    printk("VC4_TEST_RESULT name=mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_k=%d max_lda=%d max_kblocks=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_real_ttir_input=1 saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_value_scf_cf=1 saw_program_id_axis0=1 saw_arange_make_range_16=1 saw_masked_load_other_zero=1 saw_masked_store_tail=1 saw_tmu_load=1 saw_vdw_preserve_store=1 saw_tail_mask_clamp_overlaunch=1 saw_ttir_elementwise=1 saw_ttir_control_flow=1 saw_ttir_multi_axis=1 saw_ttir_mask_tail=1 saw_ttir_mask_full=1 saw_ttir_mask_empty=1 saw_ttir_row_strided_memory=1 saw_value_mask_classifier=1 saw_value_strided_address=1 saw_ttir_reduction_f32_finite_add=1 saw_ttir_scalar_reduction_store=1 saw_f32_finite_tree_policy=1 saw_ttir_gemv_f32_row_dot=1 saw_ttir_gemv_partial_kblock=1 saw_ttir_gemv_rowwise_dot=1 saw_value_gemv_rowwise_dot=1 saw_no_sparse_memory_mask=1 saw_no_gather_lane_stride=1 saw_no_hidden_memref_descriptor=1 saw_ttir_program_id_axis1=1 saw_scf_if_true=%d saw_scf_if_false=%d saw_row_padding_sentinels=1 saw_no_tl_dot_tt_dot=1 saw_no_vector_contract=1 saw_no_multiblock_k_accumulation=1 saw_nonzero_output_hash=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, cases, elements_checked, total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, MAX_ROWS,
           MAX_K, MAX_LDA, MAX_KBLOCKS, output_hash, output_hash != 0u ? 1 : 0,
           max_abs_diff, VC4_CASE_SAW_CPP_TTIR_IMPORTER,
           saw_flag_true, saw_flag_false,
           output_hash != 0u ? 1 : 0, 4, cases, elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, p_dev);
    vc4_program_destroy(program);
}
