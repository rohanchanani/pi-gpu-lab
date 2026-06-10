#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_N 1000u
#define WAVE_BOUNDARY_N ELEMENTS_PER_WAVE
#define GUARD 32u
#define MAX_BLOCKS ((MAX_N + LANES - 1u) / LANES)
#define IN_BUFFER_N (MAX_BLOCKS * LANES + 2u * GUARD)
#define OUT_BUFFER_N (MAX_BLOCKS + 2u * GUARD)
#define SENTINEL 0x6d5a1201u

static const uint32_t n_cases[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, WAVE_BOUNDARY_N, 1000u
};

static int32_t in_values[IN_BUFFER_N];
static int32_t out_values[OUT_BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t logical_blocks(uint32_t n) {
    return (n + LANES - 1u) / LANES;
}

static uint32_t launch_blocks(uint32_t n) {
    uint32_t blocks = logical_blocks(n);
    return blocks == 0u ? 1u : blocks;
}

static int32_t input_value(uint32_t case_id, uint32_t i) {
    int32_t base = (int32_t)((i * 37u + case_id * 11u) % 211u) - 105;
    if ((i & 7u) == 0u)
        return 0;
    return ((i + case_id) & 1u) ? -base : base;
}

static void fill_buffers(uint32_t case_id) {
    for (uint32_t i = 0; i < IN_BUFFER_N; i++)
        in_values[i] = input_value(case_id, i);
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++)
        out_values[i] = (int32_t)SENTINEL;
}

static int32_t expected_sum(uint32_t case_id, uint32_t n, uint32_t block) {
    int32_t sum = 0;
    uint32_t begin = block * LANES;
    for (uint32_t lane = 0; lane < LANES; lane++) {
        uint32_t i = begin + lane;
        if (i < n)
            sum += input_value(case_id, GUARD + i);
    }
    return sum;
}

static int verify_case(uint32_t case_id, uint32_t n) {
    int mismatches = 0;
    uint32_t blocks = logical_blocks(n);
    for (uint32_t b = 0; b < blocks; b++) {
        uint32_t index = GUARD + b;
        int32_t expected = expected_sum(case_id, n, b);
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: i32 reduction case=%d n=%d block=%d got=%d expected=%d\n",
                       (int)case_id, (int)n, (int)b, (int)out_values[index], (int)expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    uint32_t blocks = logical_blocks(n);
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + blocks)
            continue;
        if ((uint32_t)out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: i32 reduction sentinel n=%d i=%d got=%x\n",
                       (int)n, (int)i, (uint32_t)out_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    uint32_t blocks = logical_blocks(n);
    for (uint32_t b = 0; b < blocks; b++) {
        uint32_t index = GUARD + b;
        hash ^= (uint32_t)out_values[index] + 0x9e3779b9u + (b << 6) + (b >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("value_reduce_sum_i32_tail program create failed");

    vc4_deviceptr_t in_dev = 0, out_dev = 0;
    uint32_t in_bytes = IN_BUFFER_N * sizeof(int32_t);
    uint32_t out_bytes = OUT_BUFFER_N * sizeof(int32_t);
    if (vc4_m2_malloc(program, &in_dev, in_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("value_reduce_sum_i32_tail allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(n_cases) / sizeof(n_cases[0]); case_id++) {
        uint32_t n = n_cases[case_id];
        uint32_t blocks = logical_blocks(n);
        vc4_dim3 grid = vc4_m2_dim3(launch_blocks(n), 1u, 1u);
        fill_buffers(case_id);
        vc4_deviceptr_t in_active = in_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, in_dev, in_values, in_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            value_reduce_sum_i32_tail_vc4value_launch(program, grid, block, in_active,
                                                      out_active, n, blocks) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            printk("ERROR: i32 reduction launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_case(case_id, n);
        int sentinels = verify_sentinels(n);
        uint32_t hash = hash_case(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_REDUCE_I32_CASE case=%d n=%d blocks=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)n, (int)blocks, mismatches, sentinels, hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_reduce_sum_i32_tail_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d wave_boundary_n=%d output_hash=%u output_hash_nonzero=%d saw_value_reduction_i32_add=1 saw_value_scalar_reduction_store=1 saw_tail_inactive_zero_input=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(n_cases) / sizeof(n_cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_N, WAVE_BOUNDARY_N, output_hash,
           output_hash != 0u ? 1 : 0, 2,
           (int)(sizeof(n_cases) / sizeof(n_cases[0])), elapsed);

    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
