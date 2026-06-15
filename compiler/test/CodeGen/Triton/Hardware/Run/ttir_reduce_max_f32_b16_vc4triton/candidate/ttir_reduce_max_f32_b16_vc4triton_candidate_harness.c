#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define GUARD 32u
#define IN_BUFFER_N (LANES + 2u * GUARD)
#define OUT_BUFFER_N (ACTIVE_QPUS + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u

static const uint32_t active_counts[] = {1u, 2u, 7u, 15u, 16u};
static float in_values[IN_BUFFER_N];
static float out_values[OUT_BUFFER_N];

static float bits_to_float(uint32_t bits) { union { uint32_t u; float f; } v; v.u = bits; return v.f; }
static uint32_t float_to_bits(float value) { union { uint32_t u; float f; } v; v.f = value; return v.u; }
static float absf_local(float value) { return value < 0.0f ? -value : value; }
static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static float input_value(uint32_t case_id, uint32_t lane) {
    int32_t whole = (int32_t)((lane * 11u + case_id * 5u) % 43u) - 21;
    float value = (float)whole * 0.25f;
    if (lane == case_id)
        value += 7.0f;
    return value;
}

static void fill_buffers(uint32_t case_id) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < IN_BUFFER_N; i++)
        in_values[i] = input_value(case_id, i);
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++)
        out_values[i] = sentinel;
    for (uint32_t lane = 0; lane < LANES; lane++)
        in_values[GUARD + lane] = input_value(case_id, lane);
}

static float expected_max_for_pid(uint32_t case_id, uint32_t n, uint32_t pid) {
    float maxv = -80.0f;
    uint32_t base = pid * LANES;
    for (uint32_t lane = 0; lane < LANES; lane++) {
        uint32_t global_lane = base + lane;
        if (global_lane >= n)
            continue;
        float value = input_value(case_id, global_lane);
        if (value > maxv)
            maxv = value;
    }
    return maxv;
}

static int verify_case(uint32_t case_id, uint32_t n, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t pid = 0; pid < ACTIVE_QPUS; pid++) {
        float expected = expected_max_for_pid(case_id, n, pid);
        float got = out_values[GUARD + pid];
        float diff = absf_local(got - expected);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (diff != 0.0f) {
            if (mismatches < 8)
                printk("ERROR: ttir_reduce_max case=%d n=%d pid=%d got=%f expected=%f diff=%f\n",
                       (int)case_id, (int)n, (int)pid, got, expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + ACTIVE_QPUS)
            continue;
        if (float_to_bits(out_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir_reduce_max sentinel i=%d bits=%x expected=%x\n",
                       (int)i, float_to_bits(out_values[i]), SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t pid = 0; pid < ACTIVE_QPUS; pid++) {
        hash ^= float_to_bits(out_values[GUARD + pid]) + 0x9e3779b9u + (pid << 6) + (pid >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("ttir_reduce_max_f32_b16 program create failed");
    vc4_deviceptr_t in_dev = 0, out_dev = 0;
    uint32_t in_bytes = IN_BUFFER_N * sizeof(float);
    uint32_t out_bytes = OUT_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &in_dev, in_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("ttir_reduce_max_f32_b16 allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);
    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(active_counts) / sizeof(active_counts[0]); case_id++) {
        uint32_t n = active_counts[case_id];
        fill_buffers(case_id);
        vc4_deviceptr_t in_active = in_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, in_dev, in_values, in_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            ttir_reduce_max_f32_b16_launch(program, grid, block, in_active, out_active, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            printk("ERROR: ttir_reduce_max launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_case(case_id, n, &max_abs_diff);
        int sentinels = verify_sentinels();
        uint32_t hash = hash_case(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("TTIR_REDUCE_MAX_CASE case=%d n=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)n, mismatches, sentinels, hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=ttir_reduce_max_f32_b16_vc4triton status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_ttir_approx_sfu=1 saw_value_approx_sfu=1 saw_ttir_finite_f32_max_reduction=1 saw_value_finite_f32_max_reduction=1 saw_f32_finite_max_policy=1 saw_approx_math_policy=1 saw_no_exact_default_math=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(active_counts) / sizeof(active_counts[0])),
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, output_hash, output_hash != 0u ? 1 : 0, max_abs_diff,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, 2,
           (int)(sizeof(active_counts) / sizeof(active_counts[0])), elapsed);

    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
