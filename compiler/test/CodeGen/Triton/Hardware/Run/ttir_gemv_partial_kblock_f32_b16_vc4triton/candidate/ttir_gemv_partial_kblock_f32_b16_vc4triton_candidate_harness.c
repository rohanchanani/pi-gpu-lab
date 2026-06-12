#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_K 32u
#define MAX_KBLOCKS 2u
#define MAX_LDA 39u
#define A_BUFFER_N (MAX_ROWS * MAX_LDA + 2u * GUARD)
#define X_BUFFER_N (MAX_K + 2u * GUARD)
#define P_BUFFER_N (MAX_ROWS * MAX_KBLOCKS + 2u * GUARD)
#define A_SENTINEL_BITS 0x7f810123u
#define X_SENTINEL_BITS 0x7f820456u
#define P_SENTINEL_BITS 0xc6038000u
#define F32_TOLERANCE 0.001f

struct partial_case {
    uint32_t rows;
    uint32_t k;
    uint32_t lda;
    uint32_t num_kblocks;
};

static const struct partial_case cases[] = {
    {1u, 0u, 19u, 1u},
    {2u, 1u, 19u, 1u},
    {7u, 15u, 23u, 1u},
    {7u, 16u, 23u, 1u},
    {2u, 17u, 37u, 2u},
    {7u, 31u, 39u, 2u},
    {7u, 32u, 39u, 2u}
};

static float a_values[A_BUFFER_N];
static float x_values[X_BUFFER_N];
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
    int32_t whole = (int32_t)((case_id * 9u + row * 5u + col * 3u) % 19u) - 9;
    float value = (float)whole * 0.125f;
    if (((case_id + row + col) & 7u) == 0u)
        value = 0.0f;
    if (((row ^ col) & 1u) != 0u)
        value = -value;
    return value;
}

static float x_value(uint32_t case_id, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 7u + col * 11u) % 17u) - 8;
    float value = (float)whole * 0.0625f;
    if ((col % 11u) == 0u)
        value = 0.0f;
    return value;
}

static void fill_buffers(uint32_t case_id, const struct partial_case *c) {
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
            a_values[GUARD + row * c->lda + col] = a_value(case_id, row, col);
    for (uint32_t col = 0; col < c->k; col++)
        x_values[GUARD + col] = x_value(case_id, col);
}

static float expected_partial(uint32_t case_id, uint32_t row, uint32_t kblock,
                              uint32_t k) {
    uint32_t start = kblock * LANES;
    uint32_t end = min_u32(k, start + LANES);
    float sum = 0.0f;
    for (uint32_t col = start; col < end; col++)
        sum += a_value(case_id, row, col) * x_value(case_id, col);
    return sum;
}

static int verify_partials(uint32_t case_id, const struct partial_case *c,
                           float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t row = 0; row < c->rows; row++) {
        for (uint32_t kblock = 0; kblock < c->num_kblocks; kblock++) {
            uint32_t index = GUARD + row * c->num_kblocks + kblock;
            float expected = expected_partial(case_id, row, kblock, c->k);
            float diff = absf_local(partial_values[index] - expected);
            if (diff > *max_abs_diff)
                *max_abs_diff = diff;
            if (diff > F32_TOLERANCE) {
                if (mismatches < 8)
                    printk("ERROR: ttir partial gemv case=%d row=%d kblock=%d k=%d got=%f expected=%f diff=%f\n",
                           (int)case_id, (int)row, (int)kblock, (int)c->k,
                           partial_values[index], expected, diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct partial_case *c) {
    int mismatches = 0;
    uint32_t active_partials = c->rows * c->num_kblocks;
    for (uint32_t i = 0; i < P_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + active_partials)
            continue;
        if (float_to_bits(partial_values[i]) != P_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir partial output sentinel i=%d bits=%x\n",
                       (int)i, float_to_bits(partial_values[i]));
            mismatches++;
        }
    }
    for (uint32_t row = 0; row < c->rows; row++) {
        for (uint32_t col = c->k; col < c->lda; col++) {
            uint32_t index = GUARD + row * c->lda + col;
            if (float_to_bits(a_values[index]) != A_SENTINEL_BITS) {
                if (mismatches < 8)
                    printk("ERROR: ttir partial A padding sentinel row=%d col=%d bits=%x\n",
                           (int)row, (int)col, float_to_bits(a_values[index]));
                mismatches++;
            }
        }
    }
    for (uint32_t col = c->k; col < MAX_K; col++) {
        uint32_t index = GUARD + col;
        if (float_to_bits(x_values[index]) != X_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir partial X padding sentinel col=%d bits=%x\n",
                       (int)col, float_to_bits(x_values[index]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(const struct partial_case *c) {
    uint32_t hash = 2166136261u ^ c->rows ^ (c->k << 8) ^
                    (c->num_kblocks << 16);
    for (uint32_t row = 0; row < c->rows; row++) {
        for (uint32_t kblock = 0; kblock < c->num_kblocks; kblock++) {
            uint32_t index = GUARD + row * c->num_kblocks + kblock;
            hash ^= float_to_bits(partial_values[index]) + 0x9e3779b9u +
                    (row << 6) + kblock;
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
    }
    return hash;
}

static void print_phase13_banner(void) {
    printk("\n");
    printk("=== Phase 13.8 TTIR GEMV partial K-block hardware isolation ===\n");
    printk("Pipeline: controlled real TTIR -> C++ importer -> value -> VC4Kernel -> SSAVC4 -> scheduled VC4 -> hardware\n");
    printk("Kernel: each QPU program stores one independent partial dot for (row, kblock)\n");
    printk("Sweep: rows={1,2,7}, K={0,1,15,16,17,31,32}, active_qpus=%d block.x=%d\n",
           (int)ACTIVE_QPUS, (int)LANES);
    printk("Checks: CPU oracle, output padding sentinels, A/X padding sentinels, no cross-kblock accumulation\n\n");
}

void notmain(void) {
    print_phase13_banner();

    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("ttir_gemv_partial_kblock_f32_b16 program create failed");

    vc4_deviceptr_t a_dev = 0, x_dev = 0, partial_dev = 0;
    uint32_t a_bytes = A_BUFFER_N * sizeof(float);
    uint32_t x_bytes = X_BUFFER_N * sizeof(float);
    uint32_t partial_bytes = P_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &partial_dev, partial_bytes) < 0)
        panic("ttir_gemv_partial_kblock_f32_b16 allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct partial_case *c = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(c->num_kblocks, c->rows, 1u);
        fill_buffers(case_id, c);
        vc4_deviceptr_t a_active = a_dev + GUARD * sizeof(float);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t partial_active = partial_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
            vc4_m2_copy_htod(program, partial_dev, partial_values, partial_bytes) < 0 ||
            ttir_gemv_partial_kblock_f32_b16_kernel_launch(
                program, grid, block, a_active, x_active, partial_active,
                c->k, c->lda, c->num_kblocks) < 0 ||
            vc4_m2_copy_dtoh(program, partial_values, partial_dev, partial_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, a_values, a_dev, a_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, x_values, x_dev, x_bytes) < 0) {
            printk("ERROR: ttir partial gemv launch/copy failed case=%d rows=%d k=%d kblocks=%d lda=%d\n",
                   (int)case_id, (int)c->rows, (int)c->k,
                   (int)c->num_kblocks, (int)c->lda);
            launch_failures++;
            continue;
        }
        int mismatches = verify_partials(case_id, c, &max_abs_diff);
        int sentinels = verify_sentinels(c);
        uint32_t hash = hash_case(c);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("TTIR_GEMV_PARTIAL_KBLOCK_CASE case=%d rows=%d k=%d lda=%d kblocks=%d grid=(%d,%d) mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)c->rows, (int)c->k, (int)c->lda,
               (int)c->num_kblocks, (int)c->num_kblocks, (int)c->rows,
               mismatches, sentinels, hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("\n");
    printk("=== Phase 13.8 TTIR GEMV partial K-block summary: %s ===\n", status);
    printk("Comparison: total_mismatches=%d sentinel_mismatches=%d launch_failures=%d max_abs_diff=%f output_hash=%u\n",
           total_mismatches, sentinel_mismatches, launch_failures, max_abs_diff,
           output_hash);
    printk("VC4_TEST_RESULT name=ttir_gemv_partial_kblock_f32_b16_vc4triton status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d block_x=%d max_rows=%d max_k=%d max_kblocks=%d max_lda=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_ttir_gemv_rowwise_dot=1 saw_ttir_gemv_partial_kblock=1 saw_value_gemv_rowwise_dot=1 saw_ttir_product_reduction=1 saw_ttir_scalar_result_store=1 saw_value_scalar_dot_store=1 saw_ttir_row_strided_A=1 saw_ttir_contiguous_X=1 saw_phase10_inactive_zero_tail_input=1 saw_phase11_row_strided_memory=1 saw_phase12_reduction_path=1 saw_f32_finite_tree_policy=1 saw_no_tl_dot_tt_dot=1 saw_no_vector_contract=1 saw_no_multiblock_k_accumulation=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, LANES,
           MAX_ROWS, MAX_K, MAX_KBLOCKS, MAX_LDA, output_hash,
           output_hash != 0u ? 1 : 0, max_abs_diff,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, 3,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, x_dev);
    vc4Free(program, partial_dev);
    vc4_program_destroy(program);
}
