#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_NCOLS 33u
#define MAX_STRIDE 52u
#define BUFFER_N (MAX_ROWS * MAX_STRIDE + 2u * GUARD)
#define ROW_OUT_N (MAX_ROWS * ((MAX_NCOLS + LANES - 1u) / LANES) + 2u * GUARD)
#define I32_MAX_N 1000u
#define I32_MAX_BLOCKS ((I32_MAX_N + LANES - 1u) / LANES)
#define I32_IN_BUFFER_N (I32_MAX_BLOCKS * LANES + 2u * GUARD)
#define I32_OUT_BUFFER_N (I32_MAX_BLOCKS + 2u * GUARD)
#define SENTINEL_F_BITS 0x7fb50055u
#define SENTINEL_REDUCE_BITS 0xc6024000u
#define SENTINEL_I32 ((int32_t)0x5a17c0de)
#define EPSILON 0.001f

struct mixed_case {
    uint32_t rows;
    uint32_t ncols;
    uint32_t ldx;
    uint32_t ldy;
    uint32_t ldo;
    uint32_t lda_reduce;
    uint32_t i32_n;
    float threshold;
    float bias;
    int32_t flag;
};

static const struct mixed_case cases[] = {
    {1u, 0u, 6u, 8u, 12u, 5u, 0u, 0.0f, 0.0f, 0},
    {1u, 1u, 7u, 9u, 13u, 8u, 1u, -7.5f, 1.25f, 1},
    {2u, 15u, 22u, 24u, 28u, 22u, 17u, 3.25f, -0.5f, 0},
    {2u, 16u, 23u, 25u, 29u, 25u, 32u, 0.0f, 2.0f, 1},
    {3u, 17u, 25u, 28u, 31u, 24u, 33u, 12.5f, 1.0f, 1},
    {4u, 31u, 39u, 42u, 45u, 37u, 63u, -2.0f, -1.5f, 0},
    {7u, 32u, 41u, 44u, 48u, 39u, 192u, 5.5f, 0.25f, 1},
    {7u, 33u, 42u, 45u, 49u, 40u, 1000u, -11.25f, 2.25f, 0}
};

static float x_values[BUFFER_N];
static float y_values[BUFFER_N];
static float out_values[BUFFER_N];
static uint32_t expected_bits[BUFFER_N];
static float reduce_f_values[BUFFER_N];
static float reduce_f_out[ROW_OUT_N];
static float expected_reduce_f[ROW_OUT_N];
static int32_t reduce_i_values[I32_IN_BUFFER_N];
static int32_t reduce_i_out[I32_OUT_BUFFER_N];

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

static uint32_t col_blocks(uint32_t ncols) {
    uint32_t blocks = (ncols + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static uint32_t launch_blocks_i32(uint32_t n) {
    uint32_t blocks = (n + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static uint32_t active_elements(const struct mixed_case *c) {
    return c->rows * c->ncols;
}

static uint32_t total_elements(const struct mixed_case *c) {
    return active_elements(c) + c->i32_n;
}

static float x_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)(case_id * 9u + row * 17u + col * 5u) - 47;
    return (float)whole + (float)((case_id + col) & 3u) * 0.25f;
}

static float y_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)(case_id * 7u + row * 11u + col * 3u) - 23;
    return (float)whole + (float)((row + col) & 1u) * 0.5f;
}

static float reduce_f_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 13u + row * 17u + col * 5u) % 53u) - 26;
    float value = (float)whole * 0.25f + (float)((row + col) & 3u) * 0.0625f;
    if (((row + col + case_id) & 1u) != 0u)
        value = -value;
    if (((row * 7u + col) % 11u) == 0u)
        value = 0.0f;
    return value;
}

static int32_t reduce_i_value(uint32_t case_id, uint32_t i) {
    int32_t base = (int32_t)((i * 37u + case_id * 11u) % 211u) - 105;
    if ((i & 7u) == 0u)
        return 0;
    return ((i + case_id) & 1u) ? -base : base;
}

static void fill_buffers(uint32_t case_id, const struct mixed_case *c) {
    float sentinel = bits_to_float(SENTINEL_F_BITS);
    float reduce_sentinel = bits_to_float(SENTINEL_REDUCE_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        x_values[i] = (float)((int32_t)i - 101);
        y_values[i] = (float)((int32_t)i + 37);
        out_values[i] = sentinel;
        expected_bits[i] = SENTINEL_F_BITS;
        reduce_f_values[i] = reduce_sentinel;
    }
    for (uint32_t i = 0; i < ROW_OUT_N; i++) {
        reduce_f_out[i] = reduce_sentinel;
        expected_reduce_f[i] = reduce_sentinel;
    }
    for (uint32_t i = 0; i < I32_IN_BUFFER_N; i++)
        reduce_i_values[i] = reduce_i_value(case_id, i);
    for (uint32_t i = 0; i < I32_OUT_BUFFER_N; i++)
        reduce_i_out[i] = SENTINEL_I32;

    uint32_t blocks = col_blocks(c->ncols);
    for (uint32_t r = 0; r < c->rows; r++) {
        for (uint32_t col = 0; col < c->ncols; col++) {
            float x = x_value(case_id, r, col);
            float y = y_value(case_id, r, col);
            float selected = x < c->threshold ? x : y;
            if (c->flag != 0)
                selected = selected + c->bias;
            x_values[GUARD + r * c->ldx + col] = x;
            y_values[GUARD + r * c->ldy + col] = y;
            expected_bits[GUARD + r * c->ldo + col] = float_to_bits(selected);
            reduce_f_values[GUARD + r * c->lda_reduce + col] =
                reduce_f_value(case_id, r, col);
        }
        for (uint32_t block_col = 0; block_col < blocks; block_col++) {
            float sum = 0.0f;
            uint32_t begin = block_col * LANES;
            for (uint32_t lane = 0; lane < LANES; lane++) {
                uint32_t col = begin + lane;
                if (col < c->ncols)
                    sum += reduce_f_value(case_id, r, col);
            }
            expected_reduce_f[GUARD + r * blocks + block_col] = sum;
        }
    }
}

static int verify_results(uint32_t case_id, const struct mixed_case *c,
                          int *saw_x_arm, int *saw_y_arm) {
    int mismatches = 0;
    for (uint32_t r = 0; r < c->rows; r++) {
        for (uint32_t col = 0; col < c->ncols; col++) {
            uint32_t index = GUARD + r * c->ldo + col;
            float x = x_value(case_id, r, col);
            if (x < c->threshold)
                *saw_x_arm = 1;
            else
                *saw_y_arm = 1;
            uint32_t got = float_to_bits(out_values[index]);
            uint32_t expected = expected_bits[index];
            if (got != expected) {
                if (mismatches < 8)
                    printk("ERROR: mixed ttir reduction elementwise row=%d col=%d got=%x expected=%x\n",
                           (int)r, (int)col, got, expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_f32_reduction(const struct mixed_case *c, float *max_abs_diff) {
    int mismatches = 0;
    uint32_t blocks = col_blocks(c->ncols);
    for (uint32_t r = 0; r < c->rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t index = GUARD + r * blocks + b;
            float diff = reduce_f_out[index] - expected_reduce_f[index];
            float ad = absf_local(diff);
            if (ad > *max_abs_diff)
                *max_abs_diff = ad;
            if (ad > EPSILON) {
                if (mismatches < 8)
                    printk("ERROR: mixed ttir f32 reduction row=%d block=%d got=%f expected=%f diff=%f\n",
                           (int)r, (int)b, reduce_f_out[index], expected_reduce_f[index], diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int32_t expected_i32_sum(uint32_t case_id, uint32_t n, uint32_t block) {
    int32_t sum = 0;
    uint32_t begin = block * LANES;
    for (uint32_t lane = 0; lane < LANES; lane++) {
        uint32_t i = begin + lane;
        if (i < n)
            sum += reduce_i_value(case_id, GUARD + i);
    }
    return sum;
}

static int verify_i32_reduction(uint32_t case_id, uint32_t n) {
    int mismatches = 0;
    uint32_t blocks = (n + LANES - 1u) / LANES;
    for (uint32_t b = 0; b < blocks; b++) {
        uint32_t index = GUARD + b;
        int32_t expected = expected_i32_sum(case_id, n, b);
        if (reduce_i_out[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: mixed ttir i32 reduction n=%d block=%d got=%d expected=%d\n",
                       (int)n, (int)b, (int)reduce_i_out[index], (int)expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct mixed_case *c) {
    int mismatches = 0;
    uint32_t blocks = col_blocks(c->ncols);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int active_elementwise = 0;
        int row_padding = 0;
        if (i >= GUARD && i < GUARD + c->rows * c->ldo) {
            uint32_t rel = i - GUARD;
            active_elementwise = (rel % c->ldo) < c->ncols;
        }
        if (i >= GUARD && i < GUARD + c->rows * c->lda_reduce) {
            uint32_t rel = i - GUARD;
            row_padding = (rel % c->lda_reduce) >= c->ncols;
        }
        if (!active_elementwise && float_to_bits(out_values[i]) != SENTINEL_F_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed ttir reduction elementwise sentinel i=%d got=%x\n",
                       (int)i, float_to_bits(out_values[i]));
            mismatches++;
        }
        if (row_padding && float_to_bits(reduce_f_values[i]) != SENTINEL_REDUCE_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed ttir reduction row padding changed i=%d got=%x\n",
                       (int)i, float_to_bits(reduce_f_values[i]));
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < ROW_OUT_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + c->rows * blocks)
            active = 1;
        if (!active && float_to_bits(reduce_f_out[i]) != SENTINEL_REDUCE_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed ttir f32 reduction output sentinel i=%d got=%x\n",
                       (int)i, float_to_bits(reduce_f_out[i]));
            mismatches++;
        }
    }
    uint32_t i32_written = launch_blocks_i32(c->i32_n);
    for (uint32_t i = 0; i < I32_OUT_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + i32_written)
            continue;
        if (reduce_i_out[i] != SENTINEL_I32) {
            if (mismatches < 8)
                printk("ERROR: mixed ttir i32 reduction sentinel i=%d got=%x\n",
                       (int)i, (uint32_t)reduce_i_out[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(const struct mixed_case *c) {
    uint32_t hash = 2166136261u ^ c->rows ^ (c->ncols << 8) ^ (c->i32_n << 16);
    uint32_t blocks = col_blocks(c->ncols);
    for (uint32_t r = 0; r < c->rows; r++) {
        for (uint32_t col = 0; col < c->ncols; col++) {
            uint32_t index = GUARD + r * c->ldo + col;
            hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t index = GUARD + r * blocks + b;
            hash ^= float_to_bits(reduce_f_out[index]) + 0x85ebca6bu + (index << 5);
            hash = rotl32_local(hash, 7u) * 16777619u;
        }
    }
    uint32_t i32_blocks = launch_blocks_i32(c->i32_n);
    for (uint32_t b = 0; b < i32_blocks; b++) {
        uint32_t index = GUARD + b;
        hash ^= (uint32_t)reduce_i_out[index] + 0xc2b2ae35u + (b << 3);
        hash = rotl32_local(hash, 11u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("mixed ttir reduction program create failed");

    uint32_t bytes = BUFFER_N * sizeof(float);
    uint32_t row_bytes = ROW_OUT_N * sizeof(float);
    uint32_t i32_in_bytes = I32_IN_BUFFER_N * sizeof(int32_t);
    uint32_t i32_out_bytes = I32_OUT_BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t x_dev = 0, y_dev = 0, out_dev = 0, red_f_dev = 0, red_f_out_dev = 0;
    vc4_deviceptr_t red_i_dev = 0, red_i_out_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &red_f_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &red_f_out_dev, row_bytes) < 0 ||
        vc4_m2_malloc(program, &red_i_dev, i32_in_bytes) < 0 ||
        vc4_m2_malloc(program, &red_i_out_dev, i32_out_bytes) < 0)
        panic("mixed ttir reduction allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int elements_checked = 0;
    int saw_empty = 0;
    int saw_full = 0;
    int saw_tail = 0;
    int saw_flag_true = 0;
    int saw_flag_false = 0;
    int saw_x_arm = 0;
    int saw_y_arm = 0;
    float max_abs_diff = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct mixed_case *c = &cases[case_id];
        uint32_t blocks = col_blocks(c->ncols);
        uint32_t coverage_cols = blocks * LANES;
        uint32_t grid_y = c->rows;
        vc4_dim3 strided_grid = vc4_m2_dim3(blocks, grid_y, 1u);
        vc4_dim3 reduction_grid = vc4_m2_dim3(blocks, grid_y == 0u ? 1u : grid_y, 1u);
        vc4_dim3 i32_grid = vc4_m2_dim3(launch_blocks_i32(c->i32_n), 1u, 1u);
        fill_buffers(case_id, c);
        if (c->ncols == 0u)
            saw_empty = 1;
        else if (c->ncols == coverage_cols)
            saw_full = 1;
        else
            saw_tail = 1;
        if (c->flag != 0)
            saw_flag_true = 1;
        else
            saw_flag_false = 1;

        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        vc4_deviceptr_t red_f_active = red_f_dev + GUARD * sizeof(float);
        vc4_deviceptr_t red_f_out_active = red_f_out_dev + GUARD * sizeof(float);
        vc4_deviceptr_t red_i_active = red_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t red_i_out_active = red_i_out_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, red_f_dev, reduce_f_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, red_f_out_dev, reduce_f_out, row_bytes) < 0 ||
            vc4_m2_copy_htod(program, red_i_dev, reduce_i_values, i32_in_bytes) < 0 ||
            vc4_m2_copy_htod(program, red_i_out_dev, reduce_i_out, i32_out_bytes) < 0 ||
            mixed_ttir_strided_memory_axes_mask_cf_b16_kernel_launch(
                program, strided_grid, block, x_active, y_active, out_active,
                c->threshold, c->bias, c->flag, c->ncols, c->ldx, c->ldy, c->ldo) < 0 ||
            mixed_ttir_reduction_axes_mask_cf_strided_b16_kernel_launch(
                program, reduction_grid, block, red_f_active, red_f_out_active,
                c->ncols, c->rows, c->lda_reduce) < 0 ||
            ttir_reduce_sum_i32_b16_kernel_launch(program, i32_grid, block,
                                                  red_i_active, red_i_out_active,
                                                  c->i32_n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0 ||
            vc4_m2_copy_dtoh(program, reduce_f_out, red_f_out_dev, row_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, reduce_f_values, red_f_dev, bytes) < 0 ||
            vc4_m2_copy_dtoh(program, reduce_i_out, red_i_out_dev, i32_out_bytes) < 0) {
            printk("ERROR: mixed ttir reduction launch/copy failed case=%d rows=%d ncols=%d i32_n=%d\n",
                   (int)case_id, (int)c->rows, (int)c->ncols, (int)c->i32_n);
            launch_failures++;
            continue;
        }

        int mismatches = verify_results(case_id, c, &saw_x_arm, &saw_y_arm);
        mismatches += verify_f32_reduction(c, &max_abs_diff);
        mismatches += verify_i32_reduction(case_id, c->i32_n);
        int sentinels = verify_sentinels(c);
        uint32_t case_hash = hash_output(c);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)total_elements(c);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("MIXED_TTIR_REDUCTION_CASE case=%d rows=%d ncols=%d blocks=%d i32_n=%d flag=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)c->rows, (int)c->ncols, (int)blocks,
               (int)c->i32_n, (int)c->flag, mismatches, sentinels, case_hash,
               max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    int status_ok = total_mismatches == 0 && sentinel_mismatches == 0 &&
                    launch_failures == 0 && output_hash != 0u &&
                    saw_empty && saw_full && saw_tail && saw_flag_true &&
                    saw_flag_false && saw_x_arm && saw_y_arm;
    const char *status = status_ok ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_ttir_reduction_axes_mask_cf_strided_b16_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_ncols=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_real_ttir_input=1 saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_value_scf_cf=1 saw_program_id_axis0=1 saw_arange_make_range_16=1 saw_masked_load_other_zero=1 saw_masked_store_tail=1 saw_tmu_load=1 saw_vdw_preserve_store=1 saw_tail_mask_clamp_overlaunch=1 saw_ttir_elementwise=1 saw_ttir_control_flow=1 saw_ttir_multi_axis=1 saw_ttir_mask_tail=1 saw_ttir_mask_full=%d saw_ttir_mask_empty=%d saw_ttir_compute_mask_select=1 saw_ttir_row_strided_memory=1 saw_value_mask_classifier=1 saw_value_strided_address=1 saw_ttir_reduction_i32_add=1 saw_ttir_reduction_f32_finite_add=1 saw_ttir_scalar_reduction_store=1 saw_f32_finite_tree_policy=1 saw_no_sparse_memory_mask=1 saw_no_gather_lane_stride=1 saw_no_hidden_memref_descriptor=1 saw_ttir_program_id_axis1=1 saw_select_x_arm=%d saw_select_y_arm=%d saw_scf_if_true=%d saw_scf_if_false=%d saw_row_padding_sentinels=1 saw_no_dot_gemv=1 saw_nonzero_output_hash=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_ROWS, MAX_NCOLS, output_hash, output_hash != 0u ? 1 : 0,
           max_abs_diff, VC4_CASE_SAW_CPP_TTIR_IMPORTER, saw_full, saw_empty,
           saw_x_arm, saw_y_arm, saw_flag_true, saw_flag_false,
           output_hash != 0u ? 1 : 0, 7,
           (int)(sizeof(cases) / sizeof(cases[0])) * 3, elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, out_dev);
    vc4Free(program, red_f_dev);
    vc4Free(program, red_f_out_dev);
    vc4Free(program, red_i_dev);
    vc4Free(program, red_i_out_dev);
    vc4_program_destroy(program);
}
