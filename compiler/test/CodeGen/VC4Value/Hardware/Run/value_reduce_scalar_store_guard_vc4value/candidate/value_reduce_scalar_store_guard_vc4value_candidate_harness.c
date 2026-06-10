#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_BLOCKS 33u
#define GUARD 32u
#define OUT_BUFFER_N (MAX_BLOCKS + 2u * GUARD)
#define SENTINEL 0x7a5100d1u

struct case_desc {
    uint32_t blocks;
    int32_t value;
};

static const struct case_desc cases[] = {
    {0u, 101}, {0u, -77}, {1u, 5}, {17u, -1234}, {0u, 333}, {33u, 2047}
};

static int32_t out_values[OUT_BUFFER_N];

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t launch_blocks(uint32_t blocks) {
    return blocks == 0u ? 1u : blocks;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++)
        out_values[i] = (int32_t)SENTINEL;
}

static int verify_case(uint32_t blocks, int32_t value) {
    int mismatches = 0;
    for (uint32_t b = 0; b < blocks; b++) {
        uint32_t index = GUARD + b;
        if (out_values[index] != value) {
            if (mismatches < 8)
                printk("ERROR: scalar store block=%d got=%d expected=%d\n",
                       (int)b, (int)out_values[index], (int)value);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t blocks) {
    int mismatches = 0;
    for (uint32_t i = 0; i < OUT_BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + blocks)
            continue;
        if ((uint32_t)out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: scalar store sentinel blocks=%d i=%d got=%x\n",
                       (int)blocks, (int)i, (uint32_t)out_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_case(uint32_t blocks) {
    uint32_t hash = 2166136261u ^ blocks;
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
        panic("value_reduce_scalar_store_guard program create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t out_bytes = OUT_BUFFER_N * sizeof(int32_t);
    if (vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("value_reduce_scalar_store_guard allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t output_hash = 2166136261u;
    int saw_n0_sentinel_preserve = 1;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t blocks = cases[case_id].blocks;
        int32_t value = cases[case_id].value;
        vc4_dim3 grid = vc4_m2_dim3(launch_blocks(blocks), 1u, 1u);
        fill_output();
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            value_reduce_scalar_store_guard_vc4value_launch(program, grid, block,
                                                            out_active, value, blocks) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            printk("ERROR: scalar store guard launch/copy failed case=%d blocks=%d\n",
                   (int)case_id, (int)blocks);
            launch_failures++;
            continue;
        }
        int mismatches = verify_case(blocks, value);
        int sentinels = verify_sentinels(blocks);
        if (blocks == 0u && sentinels != 0)
            saw_n0_sentinel_preserve = 0;
        uint32_t hash = hash_case(blocks);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        output_hash ^= hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        printk("VALUE_REDUCE_SCALAR_STORE_GUARD_CASE case=%d blocks=%d value=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)blocks, (int)value, mismatches, sentinels, hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_n0_sentinel_preserve) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_reduce_scalar_store_guard_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_blocks=%d output_hash=%u output_hash_nonzero=%d saw_value_scalar_store_guard=1 saw_value_scalar_reduction_store=1 saw_repeat_invocation=1 saw_n0_sentinel_preserve=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES, MAX_BLOCKS,
           output_hash, output_hash != 0u ? 1 : 0, saw_n0_sentinel_preserve,
           1, (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
