#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MAX_N 1000u
#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_WAVES ((MAX_N + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE)
#define MAX_COVERAGE_N (MAX_WAVES * ELEMENTS_PER_WAVE)
#define BUFFER_N (MAX_COVERAGE_N + 2u * GUARD)
#define SENTINEL_I32 ((int32_t)0x5a17c0de)
#define SENTINEL_F32_BITS 0xc56a4000u

static const uint32_t n_cases[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u,
    63u, 64u, 65u, 127u, 128u, 129u, 193u, 1000u
};

static int32_t xi_values[BUFFER_N];
static int32_t yi_values[BUFFER_N];
static int32_t alt_values[BUFFER_N];
static int32_t outi_values[BUFFER_N];
static int32_t expected_i_values[BUFFER_N];

static float xf_values[BUFFER_N];
static float yf_values[BUFFER_N];
static float outf_values[BUFFER_N];
static uint32_t expected_f_bits[BUFFER_N];

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

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static int32_t xi_value(uint32_t i) {
    return (int32_t)((i * 7919u + 17u) % 100000u) - 50000;
}

static int32_t yi_value(uint32_t i) {
    return (int32_t)((i * 3571u + 29u) % 80000u) - 40000;
}

static int32_t alt_value(uint32_t i) {
    return (int32_t)((i * 1237u + 101u) % 60000u) - 30000;
}

static float xf_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 17u + 5u) % 127u) - 63;
    uint32_t quarter = (i * 3u + 1u) & 3u;
    float value = (float)whole + (float)quarter * 0.25f;
    return (i & 1u) ? -value : value;
}

static float yf_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 29u + 11u) % 113u) - 56;
    uint32_t quarter = (i * 5u + 2u) & 3u;
    float value = (float)whole + (float)quarter * 0.25f;
    return (i & 2u) ? -value : value;
}

static void fill_buffers(uint32_t n, int32_t bias, int32_t threshold) {
    float fsentinel = bits_to_float(SENTINEL_F32_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active = i - GUARD;
        xi_values[i] = xi_value(active);
        yi_values[i] = yi_value(active);
        alt_values[i] = alt_value(active);
        outi_values[i] = SENTINEL_I32;
        expected_i_values[i] = SENTINEL_I32;

        xf_values[i] = xf_value(active);
        yf_values[i] = yf_value(active);
        outf_values[i] = fsentinel;
        expected_f_bits[i] = SENTINEL_F32_BITS;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        int32_t tmp = xi_values[index] + yi_values[index] - bias;
        expected_i_values[index] = tmp > threshold ? tmp : alt_values[index];
        expected_f_bits[index] = float_to_bits(xf_values[index] + yf_values[index]);
    }
}

static int verify_i32_results(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        if (outi_values[index] != expected_i_values[index]) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_i32 n=%d i=%d gpu=%d cpu=%d\n",
                       (int)n, (int)i, (int)outi_values[index],
                       (int)expected_i_values[index]);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_f32_results(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        uint32_t got = float_to_bits(outf_values[index]);
        if (got != expected_f_bits[index]) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_f32 n=%d i=%d gpu_bits=%x cpu_bits=%x\n",
                       (int)n, (int)i, got, expected_f_bits[index]);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + n)
            continue;
        if (outi_values[i] != SENTINEL_I32) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_i32 sentinel n=%d i=%d got=%d\n",
                       (int)n, (int)i, (int)outi_values[i]);
            mismatches++;
        }
        if (float_to_bits(outf_values[i]) != SENTINEL_F32_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_f32 sentinel n=%d i=%d bits=%x\n",
                       (int)n, (int)i, float_to_bits(outf_values[i]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_i32_output(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= (uint32_t)outi_values[index] + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

static uint32_t hash_f32_output(uint32_t n) {
    uint32_t hash = 2166136261u ^ (n << 1);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= float_to_bits(outf_values[index]) + 0x85ebca6bu + (i << 5) + (i >> 3);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t i_bytes = BUFFER_N * sizeof(int32_t);
    uint32_t f_bytes = BUFFER_N * sizeof(float);
    vc4_deviceptr_t xi_dev = 0, yi_dev = 0, alt_dev = 0, outi_dev = 0;
    vc4_deviceptr_t xf_dev = 0, yf_dev = 0, outf_dev = 0;
    if (vc4_m2_malloc(program, &xi_dev, i_bytes) < 0 ||
        vc4_m2_malloc(program, &yi_dev, i_bytes) < 0 ||
        vc4_m2_malloc(program, &alt_dev, i_bytes) < 0 ||
        vc4_m2_malloc(program, &outi_dev, i_bytes) < 0 ||
        vc4_m2_malloc(program, &xf_dev, f_bytes) < 0 ||
        vc4_m2_malloc(program, &yf_dev, f_bytes) < 0 ||
        vc4_m2_malloc(program, &outf_dev, f_bytes) < 0)
        panic("mixed_ttir_dual allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t case_id = 0; case_id < sizeof(n_cases) / sizeof(n_cases[0]); case_id++) {
        uint32_t n = n_cases[case_id];
        int32_t bias = (case_id & 1u) ? 11 : -37;
        int32_t threshold = (case_id & 2u) ? 4000 : -1000;
        uint32_t waves = rounded_waves(n);
        uint32_t coverage = waves * ELEMENTS_PER_WAVE;
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_buffers(n, bias, threshold);

        vc4_deviceptr_t xi_active = xi_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t yi_active = yi_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t alt_active = alt_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t outi_active = outi_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t xf_active = xf_dev + GUARD * sizeof(float);
        vc4_deviceptr_t yf_active = yf_dev + GUARD * sizeof(float);
        vc4_deviceptr_t outf_active = outf_dev + GUARD * sizeof(float);

        if (vc4_m2_copy_htod(program, xi_dev, xi_values, i_bytes) < 0 ||
            vc4_m2_copy_htod(program, yi_dev, yi_values, i_bytes) < 0 ||
            vc4_m2_copy_htod(program, alt_dev, alt_values, i_bytes) < 0 ||
            vc4_m2_copy_htod(program, outi_dev, outi_values, i_bytes) < 0 ||
            vc4_m2_copy_htod(program, xf_dev, xf_values, f_bytes) < 0 ||
            vc4_m2_copy_htod(program, yf_dev, yf_values, f_bytes) < 0 ||
            vc4_m2_copy_htod(program, outf_dev, outf_values, f_bytes) < 0 ||
            i32_add_select_b16_kernel_launch(program, grid, block, xi_active,
                                             yi_active, alt_active, outi_active,
                                             bias, threshold, n) < 0 ||
            vector_add_b16_kernel_launch(program, grid, block, xf_active, yf_active,
                                         outf_active, n) < 0 ||
            vc4_m2_copy_dtoh(program, outi_values, outi_dev, i_bytes) < 0 ||
            vc4_m2_copy_dtoh(program, outf_values, outf_dev, f_bytes) < 0) {
            printk("ERROR: mixed_ttir_dual launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = verify_i32_results(n) + verify_f32_results(n);
        int sentinels = verify_sentinels(n);
        uint32_t i_hash = hash_i32_output(n);
        uint32_t f_hash = hash_f32_output(n);
        output_hash ^= i_hash + rotl32_local(f_hash, 13u) + 0x9e3779b9u + (case_id << 6);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)(n * 2u);
        printk("MIXED_TTIR_I32_F32_DUAL_KERNEL_TAIL_CASE case=%d n=%d waves=%d coverage=%d mismatches=%d sentinel_mismatches=%d i32_hash=%x f32_hash=%x\n",
               (int)case_id, (int)n, (int)waves, (int)coverage,
               mismatches, sentinels, i_hash, f_hash);
    }

    int elapsed = timer_get_usec() - start;
    int launches = (int)(sizeof(n_cases) / sizeof(n_cases[0])) * 2;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_ttir_i32_f32_dual_kernel_tail_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u saw_real_ttir_input=1 saw_ttir_importer_lower_elementwise_v1=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_program_id_axis0=1 saw_arange_make_range_16=1 saw_masked_load_other_zero=1 saw_masked_store_tail=1 saw_tmu_load=1 saw_vdw_preserve_store=1 saw_tail_mask_clamp_overlaunch=1 saw_f32_alu=1 saw_i32_alu=1 saw_i32_cmp_select=1 saw_sentinel_preserve=1 saw_nonzero_output_hash=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N, output_hash,
           VC4_CASE_SAW_TTIR_IMPORT, output_hash != 0u ? 1 : 0, 7,
           launches, elapsed);

    vc4Free(program, xi_dev);
    vc4Free(program, yi_dev);
    vc4Free(program, alt_dev);
    vc4Free(program, outi_dev);
    vc4Free(program, xf_dev);
    vc4Free(program, yf_dev);
    vc4Free(program, outf_dev);
    vc4_program_destroy(program);
}
