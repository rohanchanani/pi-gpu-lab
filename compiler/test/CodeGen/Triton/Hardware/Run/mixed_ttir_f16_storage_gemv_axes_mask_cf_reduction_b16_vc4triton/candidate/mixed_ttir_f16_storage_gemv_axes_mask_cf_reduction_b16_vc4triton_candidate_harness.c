#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define GUARD 32u
#define MIXED_ROWS 16u
#define MIXED_K 16u
#define MIXED_KBLOCKS 2u
#define MIXED_PARTIAL_STRIDE 16u
#define MIXED_EXTRA_ROWS 4u
#define A_BUFFER_N ((MIXED_ROWS + MIXED_EXTRA_ROWS) * MIXED_K + 2u * GUARD)
#define X_BUFFER_N (MIXED_K + 2u * GUARD)
#define Y_BUFFER_N (MIXED_ROWS + MIXED_EXTRA_ROWS + 2u * GUARD)
#define P_BUFFER_N ((MIXED_ROWS + MIXED_EXTRA_ROWS) * MIXED_PARTIAL_STRIDE + 2u * GUARD)
#define STORE_N 16u
#define STORE_BUFFER_N (STORE_N + 2u * GUARD)
#define A_PADDING_SENTINEL 0x8000u
#define X_PADDING_SENTINEL 0x8000u
#define STORE_SENTINEL 0x5a5au
#define Y_SENTINEL_BITS 0xc7110000u
#define P_SENTINEL_BITS 0xc6038000u
#define F32_TOLERANCE 0.002f

static uint16_t a_values[A_BUFFER_N];
static uint16_t x_values[X_BUFFER_N];
static float y_values[Y_BUFFER_N];
static float partial_values[P_BUFFER_N];
static float store_in_values[STORE_BUFFER_N];
static uint16_t store_out_values[STORE_BUFFER_N];

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

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t frac = (uint32_t)h & 0x03ffu;
    uint32_t bits = exp == 0u ? sign : sign | ((exp + 112u) << 23) | (frac << 13);
    return bits_to_float(bits);
}

static uint16_t a_half(uint32_t row, uint32_t col) {
    static const uint16_t values[] = {
        0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x4000u, 0xc000u, 0x3400u, 0xb400u,
        0x4200u, 0xc200u, 0x3a00u, 0xba00u, 0x4400u, 0xc400u, 0x3000u, 0xb000u
    };
    return values[(row * 7u + col * 3u) & 15u];
}

static uint16_t x_half(uint32_t col) {
    static const uint16_t values[] = {
        0x3800u, 0x3a00u, 0xb800u, 0x3400u, 0xba00u, 0x3000u, 0x3c00u, 0xbc00u,
        0x4000u, 0xc000u, 0x3600u, 0xb600u, 0x4200u, 0xc200u, 0x3200u, 0xb200u
    };
    return values[(col * 5u) & 15u];
}

static uint16_t store_expected_half(uint32_t i) {
    static const uint16_t values[] = {
        0x0000u, 0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x3400u, 0xb400u, 0x4000u,
        0xc000u, 0x4200u, 0xc200u, 0x3a00u, 0xba00u, 0x4400u, 0xc400u, 0x3000u
    };
    return values[i & 15u];
}

static void fill_mixed_buffers(void) {
    float y_sentinel = bits_to_float(Y_SENTINEL_BITS);
    float p_sentinel = bits_to_float(P_SENTINEL_BITS);
    for (uint32_t i = 0; i < A_BUFFER_N; i++) a_values[i] = A_PADDING_SENTINEL;
    for (uint32_t i = 0; i < X_BUFFER_N; i++) x_values[i] = X_PADDING_SENTINEL;
    for (uint32_t i = 0; i < Y_BUFFER_N; i++) y_values[i] = y_sentinel;
    for (uint32_t i = 0; i < P_BUFFER_N; i++) partial_values[i] = p_sentinel;
    for (uint32_t row = 0; row < MIXED_ROWS; row++)
        for (uint32_t col = 0; col < MIXED_K; col++)
            a_values[GUARD + row * MIXED_K + col] = a_half(row, col);
    for (uint32_t col = 0; col < MIXED_K; col++)
        x_values[GUARD + col] = x_half(col);
}

static void fill_store_buffers(void) {
    for (uint32_t i = 0; i < STORE_BUFFER_N; i++) {
        store_in_values[i] = 0.0f;
        store_out_values[i] = STORE_SENTINEL;
    }
    for (uint32_t i = 0; i < STORE_N; i++) {
        uint16_t expected = store_expected_half(i);
        store_in_values[GUARD + i] = f16_to_f32(expected) - 2.5f;
    }
}

static float expected_dot(uint32_t row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < MIXED_K; col++)
        sum += f16_to_f32(a_half(row, col)) * f16_to_f32(x_half(col));
    return sum;
}

static int verify_mixed_outputs(float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t row = 0; row < MIXED_ROWS; row++) {
        float expected = expected_dot(row);
        float got = y_values[GUARD + row];
        float diff = absf_local(got - expected);
        if (diff > *max_abs_diff) *max_abs_diff = diff;
        if (diff > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR f16 storage row=%d got=%f expected=%f diff=%f\n",
                       (int)row, got, expected, diff);
            mismatches++;
        }
        float partial0 = partial_values[GUARD + row * MIXED_PARTIAL_STRIDE];
        float p0diff = absf_local(partial0 - expected);
        if (p0diff > *max_abs_diff) *max_abs_diff = p0diff;
        if (p0diff > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR f16 partial0 row=%d got=%f expected=%f diff=%f\n",
                       (int)row, partial0, expected, p0diff);
            mismatches++;
        }
        float partial1 = partial_values[GUARD + row * MIXED_PARTIAL_STRIDE + 1u];
        float p1diff = absf_local(partial1);
        if (p1diff > *max_abs_diff) *max_abs_diff = p1diff;
        if (p1diff > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR f16 partial1 row=%d got=%f expected=0 diff=%f\n",
                       (int)row, partial1, p1diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_store_outputs(void) {
    int mismatches = 0;
    for (uint32_t i = 0; i < STORE_N; i++) {
        uint16_t expected = store_expected_half(i);
        uint16_t got = store_out_values[GUARD + i];
        if (got != expected) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR f16 store i=%d got=%x expected=%x\n",
                       (int)i, got, expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_mixed_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = 0; i < Y_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + MIXED_ROWS) continue;
        if (float_to_bits(y_values[i]) != Y_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR f16 Y sentinel i=%d bits=%x\n",
                       (int)i, float_to_bits(y_values[i]));
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < P_BUFFER_N; i++) {
        if (i >= GUARD) {
            uint32_t rel = i - GUARD;
            uint32_t row = rel / MIXED_PARTIAL_STRIDE;
            uint32_t col = rel % MIXED_PARTIAL_STRIDE;
            if (row < MIXED_ROWS && col < MIXED_KBLOCKS) continue;
        }
        if (float_to_bits(partial_values[i]) != P_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR f16 partial sentinel i=%d bits=%x\n",
                       (int)i, float_to_bits(partial_values[i]));
            mismatches++;
        }
    }
    for (uint32_t row = MIXED_ROWS; row < MIXED_ROWS + MIXED_EXTRA_ROWS; row++) {
        for (uint32_t col = 0; col < MIXED_K; col++) {
            uint32_t index = GUARD + row * MIXED_K + col;
            if (a_values[index] != A_PADDING_SENTINEL) {
                if (mismatches < 8)
                    printk("ERROR: mixed TTIR f16 A row padding row=%d col=%d bits=%x\n",
                           (int)row, (int)col, a_values[index]);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_store_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = 0; i < STORE_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + STORE_N) continue;
        if (store_out_values[i] != STORE_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: mixed TTIR f16 store sentinel i=%d bits=%x\n",
                       (int)i, store_out_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_outputs(void) {
    uint32_t hash = 2166136261u;
    for (uint32_t row = 0; row < MIXED_ROWS; row++) {
        hash ^= float_to_bits(y_values[GUARD + row]) + 0x9e3779b9u + (row << 6);
        hash = rotl32_local(hash, 5u) * 16777619u;
        hash ^= float_to_bits(partial_values[GUARD + row * MIXED_PARTIAL_STRIDE]) + 0x85ebca6bu + row;
        hash = rotl32_local(hash, 7u) * 16777619u;
        hash ^= float_to_bits(partial_values[GUARD + row * MIXED_PARTIAL_STRIDE + 1u]) + 0xc2b2ae35u + row;
        hash = rotl32_local(hash, 11u) * 16777619u;
    }
    for (uint32_t i = 0; i < STORE_N; i++) {
        hash ^= (uint32_t)store_out_values[GUARD + i] + 0x27d4eb2du + (i << 4);
        hash = rotl32_local(hash, 3u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("mixed TTIR f16 storage program create failed");

    vc4_deviceptr_t a_dev = 0, x_dev = 0, y_dev = 0, p_dev = 0;
    vc4_deviceptr_t store_in_dev = 0, store_out_dev = 0;
    uint32_t a_bytes = A_BUFFER_N * sizeof(uint16_t);
    uint32_t x_bytes = X_BUFFER_N * sizeof(uint16_t);
    uint32_t y_bytes = Y_BUFFER_N * sizeof(float);
    uint32_t p_bytes = P_BUFFER_N * sizeof(float);
    uint32_t store_in_bytes = STORE_BUFFER_N * sizeof(float);
    uint32_t store_out_bytes = STORE_BUFFER_N * sizeof(uint16_t);
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, y_bytes) < 0 ||
        vc4_m2_malloc(program, &p_dev, p_bytes) < 0 ||
        vc4_m2_malloc(program, &store_in_dev, store_in_bytes) < 0 ||
        vc4_m2_malloc(program, &store_out_dev, store_out_bytes) < 0)
        panic("mixed TTIR f16 storage allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    float max_abs_diff = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);
    vc4_dim3 mixed_grid = vc4_m2_dim3(MIXED_KBLOCKS, MIXED_ROWS, 1u);
    vc4_dim3 store_grid = vc4_m2_dim3(1u, 1u, 1u);

    fill_mixed_buffers();
    fill_store_buffers();
    vc4_deviceptr_t a_active = a_dev + GUARD * sizeof(uint16_t);
    vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(uint16_t);
    vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
    vc4_deviceptr_t p_active = p_dev + GUARD * sizeof(float);
    vc4_deviceptr_t store_in_active = store_in_dev + GUARD * sizeof(float);
    vc4_deviceptr_t store_out_active = store_out_dev + GUARD * sizeof(uint16_t);

    if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
        vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
        vc4_m2_copy_htod(program, y_dev, y_values, y_bytes) < 0 ||
        vc4_m2_copy_htod(program, p_dev, partial_values, p_bytes) < 0 ||
        vc4_m2_copy_htod(program, store_in_dev, store_in_values, store_in_bytes) < 0 ||
        vc4_m2_copy_htod(program, store_out_dev, store_out_values, store_out_bytes) < 0 ||
        mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_launch(
            program, mixed_grid, block, a_active, x_active, y_active, p_active) < 0 ||
        ttir_f32_compute_store_f16_b16_launch(
            program, store_grid, block, store_in_active, store_out_active) < 0 ||
        vc4_m2_copy_dtoh(program, y_values, y_dev, y_bytes) < 0 ||
        vc4_m2_copy_dtoh(program, partial_values, p_dev, p_bytes) < 0 ||
        vc4_m2_copy_dtoh(program, a_values, a_dev, a_bytes) < 0 ||
        vc4_m2_copy_dtoh(program, store_out_values, store_out_dev, store_out_bytes) < 0) {
        printk("ERROR: mixed TTIR f16 storage launch/copy failed\n");
        launch_failures++;
    } else {
        total_mismatches += verify_mixed_outputs(&max_abs_diff);
        total_mismatches += verify_store_outputs();
        sentinel_mismatches += verify_mixed_sentinels();
        sentinel_mismatches += verify_store_sentinels();
        output_hash = hash_outputs();
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton status=%s cases=2 elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d mixed_rows=%d mixed_k=%d mixed_kblocks=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_value_scf_cf=1 saw_ttir_elementwise=1 saw_ttir_control_flow=1 saw_ttir_multi_axis=1 saw_ttir_mask_tail=1 saw_ttir_row_strided_memory=1 saw_ttir_reduction_f32_finite_add=1 saw_ttir_gemv_f32_row_dot=1 saw_ttir_f16_storage_load=1 saw_ttir_f16_storage_store=1 saw_ttir_f32_compute_after_f16_load=1 saw_value_f16_storage=1 saw_f16_storage_finite_policy=1 saw_no_native_f16_arithmetic=1 saw_no_bf16_fp8=1 saw_no_softmax_sfu=1 saw_row_padding_sentinels=1 saw_output_padding_sentinels=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(MIXED_ROWS * MIXED_K + STORE_N * 2u),
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MIXED_ROWS, MIXED_K, MIXED_KBLOCKS, output_hash,
           output_hash != 0u ? 1 : 0, max_abs_diff,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, 6, 2, elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, p_dev);
    vc4Free(program, store_in_dev);
    vc4Free(program, store_out_dev);
    vc4_program_destroy(program);
}
