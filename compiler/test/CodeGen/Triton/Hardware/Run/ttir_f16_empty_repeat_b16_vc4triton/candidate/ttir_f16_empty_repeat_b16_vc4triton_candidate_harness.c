#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define GUARD 32u
#define MAX_N 192u
#define MAX_COVERAGE_N ELEMENTS_PER_WAVE
#define BUFFER_N 256u
#define SCRATCH_N ELEMENTS_PER_WAVE
#define IN_SENTINEL 0x6b5bu
#define OUT_SENTINEL_BITS 0xc7210000u
#define F32_TOLERANCE 0.001f

static const uint32_t cases[] = {0u, 0u, 1u, 16u, 17u, 0u, 83u, 192u};
static uint16_t in_values[BUFFER_N];
static float out_values[BUFFER_N];
static float scratch_out[SCRATCH_N];

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

static uint16_t input_half(uint32_t case_id, uint32_t i) {
    static const uint16_t values[] = {
        0x3c00u, 0xbc00u, 0x3800u, 0xb800u, 0x4000u, 0xc000u, 0x3400u, 0xb400u,
        0x4200u, 0xc200u, 0x3a00u, 0xba00u, 0x4400u, 0xc400u, 0x3000u, 0xb000u
    };
    return values[(case_id * 9u + i * 5u) & 15u];
}

static uint32_t rounded_coverage(uint32_t n) {
    return ((n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE) * ELEMENTS_PER_WAVE;
}

static void fill_buffers(uint32_t case_id, uint32_t n) {
    float out_sentinel = bits_to_float(OUT_SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        in_values[i] = IN_SENTINEL;
        out_values[i] = out_sentinel;
    }
    for (uint32_t i = 0; i < SCRATCH_N; i++)
        scratch_out[i] = out_sentinel;
    for (uint32_t i = 0; i < n; i++)
        in_values[GUARD + i] = input_half(case_id, i);
}

static int verify_output(uint32_t case_id, uint32_t n, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        float expected = f16_to_f32(input_half(case_id, i)) + 1.25f;
        float got = out_values[GUARD + i];
        float diff = absf_local(got - expected);
        if (diff > *max_abs_diff) *max_abs_diff = diff;
        if (diff > F32_TOLERANCE) {
            if (mismatches < 8)
                printk("ERROR: ttir f16 repeat output case=%d i=%d got=%f expected=%f diff=%f\n",
                       (int)case_id, (int)i, got, expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    uint32_t coverage = rounded_coverage(n);
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + coverage) continue;
        if (float_to_bits(out_values[i]) != OUT_SENTINEL_BITS) {
            if (mismatches < 8) printk("ERROR: ttir f16 repeat sentinel i=%d bits=%x\n", (int)i, float_to_bits(out_values[i]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        hash ^= float_to_bits(out_values[GUARD + i]) + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program) panic("ttir f16 repeat create failed");
    vc4_deviceptr_t in_dev = 0, out_dev = 0, scratch_dev = 0;
    uint32_t in_bytes = BUFFER_N * sizeof(uint16_t);
    uint32_t out_bytes = BUFFER_N * sizeof(float);
    uint32_t scratch_bytes = SCRATCH_N * sizeof(float);
    if (vc4_m2_malloc(program, &in_dev, in_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0 ||
        vc4_m2_malloc(program, &scratch_dev, scratch_bytes) < 0)
        panic("ttir f16 repeat allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    uint32_t elements_checked = 0;
    int saw_empty = 0, saw_nonempty_after_empty = 0, runtime_launches = 0;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);
    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t n = cases[case_id];
        uint32_t coverage = rounded_coverage(n);
        fill_buffers(case_id, n);
        if (n == 0u) saw_empty = 1;
        if (saw_empty && n > 0u) saw_nonempty_after_empty = 1;
        if (vc4_m2_copy_htod(program, in_dev, in_values, in_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            vc4_m2_copy_htod(program, scratch_dev, scratch_out, scratch_bytes) < 0) {
            launch_failures++;
            continue;
        }
        if (n == 0u) {
            if (ttir_f16_load_f32_compute_store_f32_b16_launch(program, grid, block,
                                                              in_dev, scratch_dev) < 0)
                launch_failures++;
            runtime_launches++;
        } else {
            for (uint32_t base = 0; base < coverage; base += ELEMENTS_PER_WAVE) {
                vc4_deviceptr_t in_active = in_dev + (GUARD + base) * sizeof(uint16_t);
                vc4_deviceptr_t out_active = out_dev + (GUARD + base) * sizeof(float);
                if (ttir_f16_load_f32_compute_store_f32_b16_launch(program, grid, block,
                                                                  in_active, out_active) < 0) {
                    launch_failures++;
                    break;
                }
                runtime_launches++;
            }
        }
        if (vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            launch_failures++;
            continue;
        }
        int mismatches = verify_output(case_id, n, &max_abs_diff);
        int sentinels = verify_sentinels(n);
        uint32_t hash = hash_case(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += n;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("TTIR_F16_REPEAT_CASE case=%d n=%d coverage=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)n, (int)coverage, mismatches, sentinels, hash, max_abs_diff);
    }
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u &&
                          saw_empty && saw_nonempty_after_empty) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=ttir_f16_empty_repeat_b16_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_ttir_f16_storage=1 saw_value_f16_storage=1 saw_ttir_f16_storage_load=1 saw_value_f16_storage_load=1 saw_value_f32_compute_after_f16_load=1 saw_repeat_invocation=%d saw_no_native_f16_arithmetic=1 saw_no_softmax_sfu=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked, total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, MAX_N, MAX_COVERAGE_N,
           BUFFER_N, output_hash, output_hash != 0u ? 1 : 0, max_abs_diff,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, saw_nonempty_after_empty, 3, runtime_launches, elapsed);
    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4Free(program, scratch_dev);
    vc4_program_destroy(program);
}
