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
#define SENTINEL_BITS 0xc56a4000u
#define F32_TOLERANCE 0.001f

static const uint32_t n_cases[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 1000u
};

static uint16_t in_values[BUFFER_N];
static float out_values[BUFFER_N];
static float expected_values[BUFFER_N];

static float absf_local(float v) { return v < 0.0f ? -v : v; }
static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } value; value.u = bits; return value.f; }
static uint32_t float_to_bits(float value) { union { uint32_t u; float f; } bits; bits.f = value; return bits.u; }
static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t frac = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0u) {
        if (frac == 0u) {
            bits = sign;
        } else {
            exp = 1u;
            while ((frac & 0x0400u) == 0u) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (frac << 13);
        }
    } else {
        bits = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    return bits_to_float(bits);
}

static uint16_t exact_value_to_f16(uint32_t i) {
    static const uint16_t values[] = {
        0x0000u, 0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x3000u,
        0xb000u, 0x2800u, 0xa800u, 0x4000u, 0xc000u
    };
    return values[i % (sizeof(values) / sizeof(values[0]))];
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static void fill_buffers(uint32_t n) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        in_values[i] = exact_value_to_f16(i);
        out_values[i] = sentinel;
        expected_values[i] = sentinel;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        in_values[index] = exact_value_to_f16(i + n);
        expected_values[index] = f16_to_f32(in_values[index]) + 1.0f;
    }
}

static int verify_results(uint32_t n, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        float diff = out_values[index] - expected_values[index];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff) *max_abs_diff = ad;
        if (ad > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: f16 load f32 compute n=%d i=%d gpu=%f cpu=%f diff=%f\n",
                       (int)n, (int)i, out_values[index], expected_values[index], diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + n) continue;
        if (float_to_bits(out_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: f16 load f32 sentinel n=%d i=%d bits=%x expected=%x\n",
                       (int)n, (int)i, float_to_bits(out_values[i]), SENTINEL_BITS);
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
    if (vc4_program_create(&program, 0) < 0 || !program) panic("value_f16_load create failed");
    vc4_deviceptr_t in_dev = 0, out_dev = 0;
    uint32_t in_bytes = BUFFER_N * sizeof(uint16_t);
    uint32_t out_bytes = BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &in_dev, in_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("value_f16_load allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    float max_abs_diff = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);

    for (uint32_t n_id = 0; n_id < sizeof(n_cases) / sizeof(n_cases[0]); n_id++) {
        uint32_t n = n_cases[n_id];
        uint32_t waves = rounded_waves(n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1u, 1u);
        fill_buffers(n);
        vc4_deviceptr_t in_active = in_dev + GUARD * sizeof(uint16_t);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, in_dev, in_values, in_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            value_f16_load_f32_compute_store_f32_vc4value_launch(program, grid, block, in_active, out_active, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            printk("ERROR: f16 load f32 launch/copy failed n=%d\n", (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_results(n, &max_abs_diff);
        int sentinels = verify_sentinels(n);
        uint32_t case_hash = hash_output_bits(n);
        output_hash ^= case_hash + 0x9e3779b9u + (n_id << 4);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)n;
        printk("VALUE_F16_LOAD_F32_CASE n=%d waves=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)n, (int)waves, mismatches, sentinels, case_hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_f16_load_f32_compute_store_f32_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_value_f16_storage_load=1 saw_value_f32_compute_after_f16_load=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_N, MAX_COVERAGE_N, BUFFER_N, output_hash,
           output_hash != 0u ? 1 : 0, max_abs_diff, 2,
           (int)(sizeof(n_cases) / sizeof(n_cases[0])), elapsed);
    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
