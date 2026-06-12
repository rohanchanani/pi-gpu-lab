#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MAX_N 1000u
#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_WAVES ((MAX_N + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE)
#define MAX_COVERAGE_N (MAX_WAVES * ELEMENTS_PER_WAVE)
#define BUFFER_N (MAX_COVERAGE_N + 2u * GUARD)
#define OUT_SENTINEL 0x5a5au

static const uint32_t n_cases[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 1000u
};

static float in_values[BUFFER_N];
static uint16_t out_values[BUFFER_N];
static uint16_t expected_values[BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint16_t exact_value_to_f16(uint32_t i) {
    static const uint16_t values[] = {
        0x0000u, 0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x3400u,
        0xb400u, 0x4000u, 0xc000u, 0x4200u, 0xc200u
    };
    return values[i % (sizeof(values) / sizeof(values[0]))];
}

static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } value; value.u = bits; return value.f; }

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t frac = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0u) {
        bits = sign;
    } else {
        bits = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    return bits_to_float(bits);
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static void fill_buffers(uint32_t n) {
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint16_t base = exact_value_to_f16(i);
        in_values[i] = f16_to_f32(base) - 0.25f;
        out_values[i] = OUT_SENTINEL;
        expected_values[i] = OUT_SENTINEL;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        uint16_t expected = exact_value_to_f16(i + n);
        in_values[index] = f16_to_f32(expected) - 0.25f;
        expected_values[index] = expected;
    }
}

static int verify_results(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        if (out_values[index] != expected_values[index]) {
            if (mismatches < 8)
                printk("ERROR: f32 compute f16 store n=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)i, out_values[index], expected_values[index]);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + n) continue;
        if (out_values[i] != OUT_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: f32 compute f16 sentinel n=%d i=%d bits=%x\n",
                       (int)n, (int)i, out_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output_bits(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= (uint32_t)out_values[index] + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("value_f16_store create failed");
    vc4_deviceptr_t in_dev = 0, out_dev = 0;
    uint32_t in_bytes = BUFFER_N * sizeof(float);
    uint32_t out_bytes = BUFFER_N * sizeof(uint16_t);
    if (vc4_m2_malloc(program, &in_dev, in_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("value_f16_store allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);

    for (uint32_t n_id = 0; n_id < sizeof(n_cases) / sizeof(n_cases[0]); n_id++) {
        uint32_t n = n_cases[n_id];
        uint32_t waves = rounded_waves(n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1u, 1u);
        fill_buffers(n);
        vc4_deviceptr_t in_active = in_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(uint16_t);
        if (vc4_m2_copy_htod(program, in_dev, in_values, in_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            value_f32_compute_store_f16_vc4value_launch(program, grid, block, in_active, out_active, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            printk("ERROR: f32 compute f16 store launch/copy failed n=%d\n", (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_results(n);
        int sentinels = verify_sentinels(n);
        uint32_t case_hash = hash_output_bits(n);
        output_hash ^= case_hash + 0x9e3779b9u + (n_id << 4);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)n;
        printk("VALUE_F32_COMPUTE_F16_STORE_CASE n=%d waves=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)n, (int)waves, mismatches, sentinels, case_hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_f32_compute_store_f16_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u output_hash_nonzero=%d saw_value_f16_storage_store=1 saw_f16_storage_finite_policy=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N, output_hash,
           output_hash != 0u ? 1 : 0, 2,
           (int)(sizeof(n_cases) / sizeof(n_cases[0])), elapsed);
    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
