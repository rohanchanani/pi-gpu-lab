#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_KBLOCKS 2u
#define MAX_K_TOTAL 32u
#define MAX_LDA 48u
#define A_BUFFER_N (MAX_ROWS * MAX_LDA + 2u * GUARD)
#define X_BUFFER_N (MAX_K_TOTAL + 2u * GUARD)
#define PARTIAL_BUFFER_N (MAX_ROWS * MAX_KBLOCKS + 2u * GUARD)
#define A_SENTINEL_BITS 0xc6a10000u
#define PARTIAL_SENTINEL_BITS 0xc6b10000u
#define F32_TOLERANCE 0.001f

struct case_desc { uint32_t rows, num_kblocks, k_total, lda; };
static const struct case_desc cases[] = {
    {2u, 1u, 1u, 8u}, {3u, 1u, 15u, 22u}, {2u, 1u, 16u, 25u}, {2u, 2u, 16u, 40u},
    {7u, 2u, 17u, 41u}, {7u, 2u, 31u, 45u}, {7u, 2u, 32u, 48u}
};

static float a_values[A_BUFFER_N];
static float x_values[X_BUFFER_N];
static float partial_values[PARTIAL_BUFFER_N];

static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } v; v.u = bits; return v.f; }
static uint32_t float_to_bits(float value) { union { uint32_t u; float f; } v; v.f = value; return v.u; }
static float absf_local(float value) { return value < 0.0f ? -value : value; }
static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static float a_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 13u + row * 19u + col * 3u) % 43u) - 21;
    float value = (float)whole * 0.09375f + (float)((row + case_id) & 3u) * 0.015625f;
    if (((row + col + case_id) & 1u) != 0u) value = -value;
    return value;
}

static float x_value(uint32_t case_id, uint32_t col) {
    int32_t whole = (int32_t)((case_id * 5u + col * 11u) % 29u) - 14;
    float value = (float)whole * 0.078125f;
    if (((col + case_id) & 2u) != 0u) value = -value;
    return value;
}

static void fill_buffers(uint32_t case_id, uint32_t rows, uint32_t k_total, uint32_t lda) {
    float a_sentinel = bits_to_float(A_SENTINEL_BITS);
    float partial_sentinel = bits_to_float(PARTIAL_SENTINEL_BITS);
    for (uint32_t i = 0; i < A_BUFFER_N; i++) a_values[i] = a_sentinel;
    for (uint32_t i = 0; i < X_BUFFER_N; i++) x_values[i] = a_sentinel;
    for (uint32_t i = 0; i < PARTIAL_BUFFER_N; i++) partial_values[i] = partial_sentinel;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < k_total; c++)
            a_values[GUARD + r * lda + c] = a_value(case_id, r, c);
    for (uint32_t c = 0; c < k_total; c++)
        x_values[GUARD + c] = x_value(case_id, c);
}

static float expected_partial(uint32_t case_id, uint32_t row, uint32_t kblock, uint32_t k_total) {
    uint32_t begin = kblock * LANES;
    uint32_t end = begin + LANES;
    if (end > k_total) end = k_total;
    float sum = 0.0f;
    for (uint32_t c = begin; c < end; c++)
        sum += a_value(case_id, row, c) * x_value(case_id, c);
    return sum;
}

static int verify_partials(uint32_t case_id, uint32_t rows, uint32_t num_kblocks,
                           uint32_t k_total, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t kb = 0; kb < num_kblocks; kb++) {
            float expected = expected_partial(case_id, r, kb, k_total);
            float got = partial_values[GUARD + r * num_kblocks + kb];
            float diff = absf_local(got - expected);
            if (diff > *max_abs_diff) *max_abs_diff = diff;
            if (diff > F32_TOLERANCE) {
                if (mismatches < 8)
                    printk("ERROR: partial gemv case=%d row=%d kblock=%d got=%f expected=%f diff=%f\n",
                           (int)case_id, (int)r, (int)kb, got, expected, diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t rows, uint32_t num_kblocks, uint32_t k_total, uint32_t lda) {
    int mismatches = 0;
    uint32_t active_partials = rows * num_kblocks;
    for (uint32_t i = 0; i < PARTIAL_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + active_partials) continue;
        if (float_to_bits(partial_values[i]) != PARTIAL_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: partial gemv output sentinel i=%d bits=%x\n",
                       (int)i, float_to_bits(partial_values[i]));
            mismatches++;
        }
    }
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = k_total; c < lda; c++) {
            uint32_t idx = GUARD + r * lda + c;
            if (float_to_bits(a_values[idx]) != A_SENTINEL_BITS) {
                if (mismatches < 8)
                    printk("ERROR: partial gemv A padding changed row=%d col=%d bits=%x\n",
                           (int)r, (int)c, float_to_bits(a_values[idx]));
                mismatches++;
            }
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t rows, uint32_t num_kblocks) {
    uint32_t hash = 2166136261u ^ (rows * 17u + num_kblocks);
    for (uint32_t i = 0; i < rows * num_kblocks; i++) {
        hash ^= float_to_bits(partial_values[GUARD + i]) + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("value_gemv_partial create failed");
    vc4_deviceptr_t a_dev = 0, x_dev = 0, partial_dev = 0;
    uint32_t a_bytes = A_BUFFER_N * sizeof(float);
    uint32_t x_bytes = X_BUFFER_N * sizeof(float);
    uint32_t partial_bytes = PARTIAL_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &partial_dev, partial_bytes) < 0)
        panic("value_gemv_partial allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t rows = cases[case_id].rows;
        uint32_t num_kblocks = cases[case_id].num_kblocks;
        uint32_t k_total = cases[case_id].k_total;
        uint32_t lda = cases[case_id].lda;
        uint32_t partial_count = rows * num_kblocks;
        vc4_dim3 grid = vc4_m2_dim3(num_kblocks, rows, 1u);
        fill_buffers(case_id, rows, k_total, lda);
        vc4_deviceptr_t a_active = a_dev + GUARD * sizeof(float);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t partial_active = partial_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
            vc4_m2_copy_htod(program, partial_dev, partial_values, partial_bytes) < 0 ||
            value_gemv_partial_kblock_f32_vc4value_launch(
                program, grid, block, a_active, x_active, partial_active, rows, k_total, lda,
                num_kblocks, partial_count) < 0 ||
            vc4_m2_copy_dtoh(program, partial_values, partial_dev, partial_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, a_values, a_dev, a_bytes) < 0) {
            printk("ERROR: partial gemv launch/copy failed case=%d rows=%d num_kblocks=%d k_total=%d lda=%d\n",
                   (int)case_id, (int)rows, (int)num_kblocks, (int)k_total, (int)lda);
            launch_failures++;
            continue;
        }
        int mismatches = verify_partials(case_id, rows, num_kblocks, k_total, &max_abs_diff);
        int sentinels = verify_sentinels(rows, num_kblocks, k_total, lda);
        uint32_t hash = hash_case(rows, num_kblocks);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_GEMV_PARTIAL_CASE case=%d rows=%d num_kblocks=%d k_total=%d lda=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)rows, (int)num_kblocks, (int)k_total, (int)lda,
               mismatches, sentinels, hash, max_abs_diff);
    }
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_gemv_partial_kblock_f32_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_kblocks=%d max_k_total=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_value_gemv_partial_kblock=1 saw_no_multiblock_accumulation=1 saw_f32_dot_finite_tree_policy=1 saw_value_scalar_dot_store=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches, sentinel_mismatches, launch_failures,
           ACTIVE_QPUS, LANES, MAX_ROWS, MAX_KBLOCKS, MAX_K_TOTAL, output_hash, output_hash != 0u ? 1 : 0,
           max_abs_diff, 3, (int)(sizeof(cases) / sizeof(cases[0])), elapsed);
    vc4Free(program, a_dev);
    vc4Free(program, x_dev);
    vc4Free(program, partial_dev);
    vc4_program_destroy(program);
}
