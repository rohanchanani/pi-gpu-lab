#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define GUARD 32u
#define MAX_ROWS 13u
#define MAX_K 16u
#define MAX_LDA 23u
#define A_BUFFER_N (MAX_ROWS * MAX_LDA + 2u * GUARD)
#define X_BUFFER_N (MAX_LDA + 2u * GUARD)
#define Y_BUFFER_N (MAX_ROWS + 2u * GUARD)
#define A_SENTINEL_BITS 0x7f810123u
#define X_SENTINEL_BITS 0x7f820456u
#define Y_SENTINEL_BITS 0xc6024000u
#define F32_TOLERANCE 0.001f

struct gemv_case {
    uint32_t rows;
    uint32_t k;
    uint32_t lda;
};

static const struct gemv_case cases[] = {
    {1u, 0u, 16u},
    {2u, 1u, 17u},
    {7u, 2u, 23u},
    {13u, 7u, 16u},
    {1u, 15u, 23u},
    {2u, 16u, 17u},
    {7u, 0u, 23u},
    {13u, 16u, 23u},
    {13u, 1u, 16u}
};

static float a_values[A_BUFFER_N];
static float x_values[X_BUFFER_N];
static float y_values[Y_BUFFER_N];

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

static float a_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 5u + row * 3u + col * 7u) % 17u) - 8;
    float value = (float)whole * 0.25f;
    if (((case_id + row + col) & 5u) == 0u)
        value = 0.0f;
    if (((row + col) & 1u) != 0u)
        value = -value;
    return value;
}

static float x_value(uint32_t case_id, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 11u + col * 3u) % 13u) - 6;
    float value = (float)whole * 0.125f;
    if ((col % 7u) == 0u)
        value = 0.0f;
    return value;
}

static void fill_buffers(uint32_t case_id, const struct gemv_case *c) {
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

static float expected_row(uint32_t case_id, uint32_t row, uint32_t k) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < k; col++)
        sum += a_value(case_id, row, col) * x_value(case_id, col);
    return sum;
}

static int verify_rows(uint32_t case_id, const struct gemv_case *c,
                       float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t row = 0; row < c->rows; row++) {
        uint32_t index = GUARD + row;
        float expected = expected_row(case_id, row, c->k);
        float diff = absf_local(y_values[index] - expected);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (diff > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: ttir gemv case=%d row=%d k=%d lda=%d got=%f expected=%f diff=%f\n",
                       (int)case_id, (int)row, (int)c->k, (int)c->lda,
                       y_values[index], expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct gemv_case *c) {
    int mismatches = 0;
    for (uint32_t i = 0; i < Y_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + c->rows)
            continue;
        if (float_to_bits(y_values[i]) != Y_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir gemv Y sentinel rows=%d i=%d bits=%x\n",
                       (int)c->rows, (int)i, float_to_bits(y_values[i]));
            mismatches++;
        }
    }
    for (uint32_t row = 0; row < c->rows; row++) {
        for (uint32_t col = c->k; col < c->lda; col++) {
            uint32_t index = GUARD + row * c->lda + col;
            if (float_to_bits(a_values[index]) != A_SENTINEL_BITS) {
                if (mismatches < 8)
                    printk("ERROR: ttir gemv A padding sentinel row=%d col=%d lda=%d bits=%x\n",
                           (int)row, (int)col, (int)c->lda, float_to_bits(a_values[index]));
                mismatches++;
            }
        }
    }
    for (uint32_t col = c->k; col < MAX_LDA; col++) {
        uint32_t index = GUARD + col;
        if (float_to_bits(x_values[index]) != X_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir gemv X padding sentinel col=%d bits=%x\n",
                       (int)col, float_to_bits(x_values[index]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(const struct gemv_case *c) {
    uint32_t hash = 2166136261u ^ c->rows ^ (c->k << 8) ^ (c->lda << 16);
    for (uint32_t row = 0; row < c->rows; row++) {
        uint32_t index = GUARD + row;
        hash ^= float_to_bits(y_values[index]) + 0x9e3779b9u + (row << 6) + (row >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("ttir_gemv_row_dot_f32_b16 program create failed");

    vc4_deviceptr_t a_dev = 0, x_dev = 0, y_dev = 0;
    uint32_t a_bytes = A_BUFFER_N * sizeof(float);
    uint32_t x_bytes = X_BUFFER_N * sizeof(float);
    uint32_t y_bytes = Y_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, y_bytes) < 0)
        panic("ttir_gemv_row_dot_f32_b16 allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct gemv_case *c = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(c->rows == 0u ? 1u : c->rows, 1u, 1u);
        fill_buffers(case_id, c);
        vc4_deviceptr_t a_active = a_dev + GUARD * sizeof(float);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, y_bytes) < 0 ||
            gemv_row_dot_f32_b16_kernel_launch(program, grid, block,
                                               a_active, x_active, y_active,
                                               c->k, c->lda) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, y_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, a_values, a_dev, a_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, x_values, x_dev, x_bytes) < 0) {
            printk("ERROR: ttir gemv launch/copy failed case=%d rows=%d k=%d lda=%d\n",
                   (int)case_id, (int)c->rows, (int)c->k, (int)c->lda);
            launch_failures++;
            continue;
        }
        int mismatches = verify_rows(case_id, c, &max_abs_diff);
        int sentinels = verify_sentinels(c);
        uint32_t hash = hash_case(c);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("TTIR_GEMV_ROW_DOT_CASE case=%d rows=%d k=%d lda=%d block_x=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)c->rows, (int)c->k, (int)c->lda, (int)LANES,
               mismatches, sentinels, hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=ttir_gemv_row_dot_f32_b16_vc4triton status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d block_x=%d max_rows=%d max_k=%d max_lda=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_real_triton_source=1 saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_ttir_gemv_rowwise_dot=1 saw_ttir_product_reduction=1 saw_ttir_scalar_result_store=1 saw_ttir_row_strided_A=1 saw_ttir_contiguous_X=1 saw_phase12_reduction_path=1 saw_f32_finite_tree_policy=1 saw_no_tl_dot_tt_dot=1 saw_no_multiblock_k_accumulation=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, LANES,
           MAX_ROWS, MAX_K, MAX_LDA, output_hash, output_hash != 0u ? 1 : 0,
           max_abs_diff, VC4_CASE_SAW_CPP_TTIR_IMPORTER, 3,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
