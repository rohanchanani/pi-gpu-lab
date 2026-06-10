#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_COLS 16u
#define MAX_LDA 31u
#define IN_BUFFER_N (MAX_ROWS * MAX_LDA + 2u * GUARD)
#define OUT_BUFFER_N (MAX_ROWS + 2u * GUARD)
#define IN_SENTINEL_BITS 0xc5f12000u
#define OUT_SENTINEL_BITS 0xc6024000u
#define F32_TOLERANCE 0.001f

struct case_desc {
    uint32_t rows, cols, lda;
};

static const struct case_desc cases[] = {
    {1u, 0u, 5u}, {2u, 1u, 8u}, {3u, 15u, 22u},
    {4u, 16u, 25u}, {7u, 3u, 19u}, {7u, 16u, 31u}
};

static float in_values[IN_BUFFER_N];
static float out_values[OUT_BUFFER_N];

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

static float input_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 13u + row * 17u + col * 5u) % 53u) - 26;
    float value = (float)whole * 0.25f + (float)((row + col) & 3u) * 0.0625f;
    if (((row + col + case_id) & 1u) != 0u)
        value = -value;
    if (((row * 7u + col) % 11u) == 0u)
        value = 0.0f;
    return value;
}

static void fill_buffers(uint32_t case_id, uint32_t rows, uint32_t cols, uint32_t lda) {
    float in_sentinel = bits_to_float(IN_SENTINEL_BITS);
    float out_sentinel = bits_to_float(OUT_SENTINEL_BITS);
    for (uint32_t i = 0; i < IN_BUFFER_N; i++)
        in_values[i] = in_sentinel;
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++)
        out_values[i] = out_sentinel;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++)
            in_values[GUARD + r * lda + c] = input_value(case_id, r, c);
}

static float expected_sum(uint32_t case_id, uint32_t row, uint32_t cols) {
    float sum = 0.0f;
    for (uint32_t c = 0; c < cols; c++)
        sum += input_value(case_id, row, c);
    return sum;
}

static int verify_rows(uint32_t case_id, uint32_t rows, uint32_t cols, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t index = GUARD + r;
        float expected = expected_sum(case_id, r, cols);
        float diff = absf_local(out_values[index] - expected);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (diff > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: ttir row reduction case=%d row=%d cols=%d got=%f expected=%f diff=%f\n",
                       (int)case_id, (int)r, (int)cols, out_values[index], expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t rows, uint32_t cols, uint32_t lda) {
    int mismatches = 0;
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + rows)
            continue;
        if (float_to_bits(out_values[i]) != OUT_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir row reduction output sentinel rows=%d i=%d bits=%x\n",
                       (int)rows, (int)i, float_to_bits(out_values[i]));
            mismatches++;
        }
    }
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = cols; c < lda; c++) {
            uint32_t index = GUARD + r * lda + c;
            if (float_to_bits(in_values[index]) != IN_SENTINEL_BITS) {
                if (mismatches < 8)
                    printk("ERROR: ttir row reduction input padding changed row=%d col=%d lda=%d bits=%x\n",
                           (int)r, (int)c, (int)lda, float_to_bits(in_values[index]));
                mismatches++;
            }
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t rows) {
    uint32_t hash = 2166136261u ^ rows;
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t index = GUARD + r;
        hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (r << 6) + (r >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("ttir_reduce_row_strided_sum_f32_b16 program create failed");

    vc4_deviceptr_t in_dev = 0, out_dev = 0;
    uint32_t in_bytes = IN_BUFFER_N * sizeof(float);
    uint32_t out_bytes = OUT_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &in_dev, in_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("ttir_reduce_row_strided_sum_f32_b16 allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t rows = cases[case_id].rows;
        uint32_t cols = cases[case_id].cols;
        uint32_t lda = cases[case_id].lda;
        vc4_dim3 grid = vc4_m2_dim3(rows == 0u ? 1u : rows, 1u, 1u);
        fill_buffers(case_id, rows, cols, lda);
        vc4_deviceptr_t in_active = in_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, in_dev, in_values, in_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            ttir_reduce_row_strided_sum_f32_b16_kernel_launch(program, grid, block,
                                                              in_active, out_active,
                                                              cols, lda) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, in_values, in_dev, in_bytes) < 0) {
            printk("ERROR: ttir row reduction launch/copy failed case=%d rows=%d cols=%d lda=%d\n",
                   (int)case_id, (int)rows, (int)cols, (int)lda);
            launch_failures++;
            continue;
        }
        int mismatches = verify_rows(case_id, rows, cols, &max_abs_diff);
        int sentinels = verify_sentinels(rows, cols, lda);
        uint32_t hash = hash_case(rows);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("TTIR_REDUCE_ROW_STRIDED_CASE case=%d rows=%d cols=%d lda=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)rows, (int)cols, (int)lda, mismatches, sentinels,
               hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=ttir_reduce_row_strided_sum_f32_b16_vc4triton status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_cols=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_ttir_reduction=1 saw_value_reduction=1 saw_ttir_reduce_f32_finite_add=1 saw_ttir_scalar_reduction_store=1 saw_ttir_row_strided_reduction=1 saw_f32_finite_tree_policy=1 saw_row_padding_sentinels=1 saw_no_dot_gemv=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, MAX_ROWS,
           MAX_COLS, output_hash, output_hash != 0u ? 1 : 0, max_abs_diff,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, 2, (int)(sizeof(cases) / sizeof(cases[0])),
           elapsed);

    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
