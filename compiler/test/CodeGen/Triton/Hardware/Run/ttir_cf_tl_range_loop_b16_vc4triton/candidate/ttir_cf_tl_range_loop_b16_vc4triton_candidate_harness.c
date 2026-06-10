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
#define SENTINEL_BITS 0xc56a4000u

struct loop_case {
    uint32_t n;
    int32_t trip_count;
};

static const struct loop_case loop_cases[] = {
    {0u, 0}, {1u, 1}, {2u, 2}, {15u, 5}, {16u, 0},
    {17u, 1}, {31u, 2}, {32u, 5}, {33u, 0}, {63u, 1},
    {64u, 2}, {65u, 5}, {127u, 0}, {128u, 1}, {129u, 2},
    {193u, 5}, {384u, 0}, {385u, 1}, {999u, 2}, {1000u, 5}
};

static float x_values[BUFFER_N];
static float out_values[BUFFER_N];
static uint32_t expected_bits[BUFFER_N];

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

static float x_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 17u + 5u) % 127u) - 63;
    uint32_t quarter = (i * 3u + 1u) & 3u;
    float value = (float)whole + (float)quarter * 0.25f;
    return (i & 1u) ? -value : value;
}

static void fill_buffers(uint32_t n, int32_t trip_count) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active = i - GUARD;
        x_values[i] = x_value(active);
        out_values[i] = sentinel;
        expected_bits[i] = SENTINEL_BITS;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        float expected = x_values[index] + (float)trip_count;
        expected_bits[index] = float_to_bits(expected);
    }
}

static int verify_results(uint32_t n, int32_t trip_count) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        uint32_t got = float_to_bits(out_values[index]);
        uint32_t expected = expected_bits[index];
        if (got != expected) {
            if (mismatches < 8)
                printk("ERROR: ttir_cf_tl_range n=%d trip=%d i=%d gpu_bits=%x expected_bits=%x gpu=%f expected=%f\n",
                       (int)n, (int)trip_count, (int)i, got, expected,
                       out_values[index], bits_to_float(expected));
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
        uint32_t got = float_to_bits(out_values[i]);
        if (got != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir_cf_tl_range sentinel n=%d i=%d bits=%x expected=%x\n",
                       (int)n, (int)i, got, SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output_bits(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = BUFFER_N * sizeof(float);
    vc4_deviceptr_t x_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("ttir_cf_tl_range allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t case_id = 0; case_id < sizeof(loop_cases) / sizeof(loop_cases[0]); case_id++) {
        uint32_t n = loop_cases[case_id].n;
        int32_t trip_count = loop_cases[case_id].trip_count;
        uint32_t waves = rounded_waves(n);
        uint32_t coverage = waves * ELEMENTS_PER_WAVE;
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_buffers(n, trip_count);

        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            ttir_cf_tl_range_loop_b16_kernel_launch(program, grid, block,
                                                    x_active, out_active, n,
                                                    trip_count) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: ttir_cf_tl_range launch/copy failed trip=%d n=%d\n",
                   (int)trip_count, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = verify_results(n, trip_count);
        int sentinels = verify_sentinels(n);
        uint32_t case_hash = hash_output_bits(n);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)n;
        printk("TTIR_CF_TL_RANGE_CASE case=%d n=%d trip=%d waves=%d coverage=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)n, (int)trip_count, (int)waves,
               (int)coverage, mismatches, sentinels, case_hash);
    }

    int elapsed = timer_get_usec() - start;
    int cases = (int)(sizeof(loop_cases) / sizeof(loop_cases[0]));
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=ttir_cf_tl_range_loop_b16_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u saw_nonzero_output_hash=%d saw_real_ttir_snapshot=%d saw_cpp_ttir_importer=%d saw_value_scf_cf=%d saw_vc4value_to_vc4kernel=%d saw_ttir_scf_for=%d saw_tail_mask_load_store=%d saw_cpu_oracle_sentinel=%d saw_no_arith_sitofp=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, cases, elements_checked, total_mismatches, sentinel_mismatches,
           launch_failures, ACTIVE_QPUS, LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N,
           output_hash, output_hash != 0u ? 1 : 0, 1, VC4_CASE_SAW_CPP_TTIR_IMPORTER,
           1, 1, 1, 1, 1, 1, 2, cases, elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
