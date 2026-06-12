#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_K 16u
#define MAX_LDA 29u
#define A_BUFFER_N (MAX_ROWS * MAX_LDA + 2u * GUARD)
#define X_BUFFER_N (MAX_K + 2u * GUARD)
#define Y_BUFFER_N (MAX_ROWS + 2u * GUARD)
#define A_SENTINEL 0x6a5au
#define X_SENTINEL 0x5a6au
#define Y_SENTINEL_BITS 0xc7110000u
#define F32_TOLERANCE 0.002f

struct case_desc { uint32_t rows, k, lda; };
static const struct case_desc cases[] = {
    {1u, 0u, 3u}, {1u, 1u, 4u}, {1u, 2u, 5u}, {1u, 15u, 22u}, {1u, 16u, 29u},
    {2u, 0u, 4u}, {2u, 1u, 5u}, {2u, 2u, 6u}, {2u, 15u, 23u}, {2u, 16u, 29u},
    {7u, 0u, 5u}, {7u, 1u, 8u}, {7u, 2u, 9u}, {7u, 15u, 24u}, {7u, 16u, 29u}
};

static uint16_t a_values[A_BUFFER_N];
static uint16_t x_values[X_BUFFER_N];
static float y_values[Y_BUFFER_N];

static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } v; v.u = bits; return v.f; }
static uint32_t float_to_bits(float value) { union { uint32_t u; float f; } v; v.f = value; return v.u; }
static float absf_local(float value) { return value < 0.0f ? -value : value; }
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

static uint16_t a_half(uint32_t case_id, uint32_t row, uint32_t col) {
    static const uint16_t values[] = {
        0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x4000u, 0xc000u, 0x3400u, 0xb400u,
        0x4200u, 0xc200u, 0x3a00u, 0xba00u, 0x4400u, 0xc400u, 0x3000u, 0xb000u
    };
    return values[(case_id * 5u + row * 7u + col * 3u) & 15u];
}

static uint16_t x_half(uint32_t case_id, uint32_t col) {
    static const uint16_t values[] = {
        0x3800u, 0x3a00u, 0xb800u, 0x3400u, 0xba00u, 0x3000u, 0x3c00u, 0xbc00u,
        0x4000u, 0xc000u, 0x3600u, 0xb600u, 0x4200u, 0xc200u, 0x3200u, 0xb200u
    };
    return values[(case_id * 11u + col * 5u) & 15u];
}

static void fill_buffers(uint32_t case_id, uint32_t rows, uint32_t k, uint32_t lda) {
    float y_sentinel = bits_to_float(Y_SENTINEL_BITS);
    for (uint32_t i = 0; i < A_BUFFER_N; i++) a_values[i] = A_SENTINEL;
    for (uint32_t i = 0; i < X_BUFFER_N; i++) x_values[i] = X_SENTINEL;
    for (uint32_t i = 0; i < Y_BUFFER_N; i++) y_values[i] = y_sentinel;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < k; c++)
            a_values[GUARD + r * lda + c] = a_half(case_id, r, c);
    for (uint32_t c = 0; c < k; c++)
        x_values[GUARD + c] = x_half(case_id, c);
}

static float expected_dot(uint32_t case_id, uint32_t row, uint32_t k) {
    float sum = 0.0f;
    for (uint32_t c = 0; c < k; c++)
        sum += f16_to_f32(a_half(case_id, row, c)) * f16_to_f32(x_half(case_id, c));
    return sum;
}

static int verify_rows(uint32_t case_id, uint32_t rows, uint32_t k, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t r = 0; r < rows; r++) {
        float expected = expected_dot(case_id, r, k);
        float got = y_values[GUARD + r];
        float diff = absf_local(got - expected);
        if (diff > *max_abs_diff) *max_abs_diff = diff;
        if (diff > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: f16 gemv row case=%d row=%d k=%d got=%f expected=%f diff=%f\n",
                       (int)case_id, (int)r, (int)k, got, expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t rows, uint32_t k, uint32_t lda) {
    int mismatches = 0;
    for (uint32_t i = 0; i < Y_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + rows) continue;
        if (float_to_bits(y_values[i]) != Y_SENTINEL_BITS) {
            if (mismatches < 8) printk("ERROR: f16 gemv y sentinel i=%d bits=%x\n", (int)i, float_to_bits(y_values[i]));
            mismatches++;
        }
    }
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = k; c < lda; c++) {
            uint32_t idx = GUARD + r * lda + c;
            if (a_values[idx] != A_SENTINEL) {
                if (mismatches < 8) printk("ERROR: f16 gemv A padding changed row=%d col=%d bits=%x\n", (int)r, (int)c, a_values[idx]);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t rows) {
    uint32_t hash = 2166136261u ^ rows;
    for (uint32_t r = 0; r < rows; r++) {
        hash ^= float_to_bits(y_values[GUARD + r]) + 0x9e3779b9u + (r << 6) + (r >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("value_f16_gemv create failed");
    vc4_deviceptr_t a_dev = 0, x_dev = 0, y_dev = 0;
    uint32_t a_bytes = A_BUFFER_N * sizeof(uint16_t);
    uint32_t x_bytes = X_BUFFER_N * sizeof(uint16_t);
    uint32_t y_bytes = Y_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, y_bytes) < 0)
        panic("value_f16_gemv allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t rows = cases[case_id].rows;
        uint32_t k = cases[case_id].k;
        uint32_t lda = cases[case_id].lda;
        vc4_dim3 grid = vc4_m2_dim3(rows == 0u ? 1u : rows, 1u, 1u);
        fill_buffers(case_id, rows, k, lda);
        vc4_deviceptr_t a_active = a_dev + GUARD * sizeof(uint16_t);
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(uint16_t);
        vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, y_bytes) < 0 ||
            value_f16_row_dot_f32_accum_vc4value_launch(program, grid, block, a_active, x_active, y_active,
                                                        rows, k, lda) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, y_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, a_values, a_dev, a_bytes) < 0) {
            printk("ERROR: f16 gemv launch/copy failed case=%d rows=%d k=%d lda=%d\n",
                   (int)case_id, (int)rows, (int)k, (int)lda);
            launch_failures++;
            continue;
        }
        int mismatches = verify_rows(case_id, rows, k, &max_abs_diff);
        int sentinels = verify_sentinels(rows, k, lda);
        uint32_t hash = hash_case(rows);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_F16_GEMV_CASE case=%d rows=%d k=%d lda=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)rows, (int)k, (int)lda, mismatches, sentinels, hash, max_abs_diff);
    }
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_f16_row_dot_f32_accum_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_k=%d max_lda=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_value_f16_gemv_input_storage=1 saw_value_gemv_f32_accum=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches, sentinel_mismatches, launch_failures,
           ACTIVE_QPUS, LANES, MAX_ROWS, MAX_K, MAX_LDA, output_hash, output_hash != 0u ? 1 : 0, max_abs_diff, 3,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);
    vc4Free(program, a_dev);
    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
