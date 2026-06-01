#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define VECTOR_STORE_SMOKE_VC4KERNEL_LANES 16u
#define VECTOR_STORE_SMOKE_VC4KERNEL_ACTIVE_QPUS 1u
#define VECTOR_STORE_SMOKE_VC4KERNEL_MAX_N 32u
#define VECTOR_STORE_SMOKE_VC4KERNEL_GUARD 16u
#define VECTOR_STORE_SMOKE_VC4KERNEL_BUFFER_N (VECTOR_STORE_SMOKE_VC4KERNEL_MAX_N + VECTOR_STORE_SMOKE_VC4KERNEL_GUARD)
#define VECTOR_STORE_SMOKE_VC4KERNEL_SENTINEL 0xdeadbeefu
#define VECTOR_STORE_SMOKE_VC4KERNEL_TAG 0x51000000u

struct vector_store_smoke_vc4kernel_case {
    uint32_t offset_elems;
    uint32_t base_value;
};

static const struct vector_store_smoke_vc4kernel_case cases[] = {
    {0u, 0u},
    {16u, 0x100u},
};

static uint32_t out_values[VECTOR_STORE_SMOKE_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < VECTOR_STORE_SMOKE_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = VECTOR_STORE_SMOKE_VC4KERNEL_SENTINEL;
}

static uint32_t expected_value(uint32_t base_value, uint32_t lane) {
    return VECTOR_STORE_SMOKE_VC4KERNEL_TAG + base_value + lane;
}

static int verify_active(uint32_t offset_elems, uint32_t base_value) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < VECTOR_STORE_SMOKE_VC4KERNEL_LANES; lane++) {
        uint32_t index = offset_elems + lane;
        uint32_t expected = expected_value(base_value, lane);
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: vector_store_vc4kernel active index=%d gpu=%x expected=%x\n", (int)index, out_values[index], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t offset_elems) {
    int mismatches = 0;
    uint32_t active_end = offset_elems + VECTOR_STORE_SMOKE_VC4KERNEL_LANES;
    for (uint32_t i = 0; i < VECTOR_STORE_SMOKE_VC4KERNEL_BUFFER_N; i++) {
        if (i >= offset_elems && i < active_end)
            continue;
        if (out_values[i] != VECTOR_STORE_SMOKE_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: vector_store_vc4kernel sentinel index=%d gpu=%x expected=%x\n", (int)i, out_values[i], VECTOR_STORE_SMOKE_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t offset_elems) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < VECTOR_STORE_SMOKE_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[offset_elems + lane] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = VECTOR_STORE_SMOKE_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("vector_store_smoke_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(VECTOR_STORE_SMOKE_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 vector_store_smoke_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t offset_elems = cases[case_id].offset_elems;
        uint32_t base_value = cases[case_id].base_value;
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            vector_store_smoke_vc4kernel_launch(program, grid, block, out_dev, offset_elems, base_value) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: vector_store_smoke_vc4kernel launch/copy failed case=%d offset_elems=%d\n", (int)case_id, (int)offset_elems);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(offset_elems, base_value);
        int sentinels = verify_sentinels(offset_elems);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        int checksum = checksum_low16(offset_elems);
        checksum_accum += checksum;
        printk("VECTOR_STORE_SMOKE_VC4KERNEL_CASE case=%d offset_elems=%d base_value=%x mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)offset_elems, base_value, mismatches, sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=vector_store_smoke_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches, sentinel_mismatches, launch_failures,
           VECTOR_STORE_SMOKE_VC4KERNEL_ACTIVE_QPUS, VECTOR_STORE_SMOKE_VC4KERNEL_LANES, VECTOR_STORE_SMOKE_VC4KERNEL_MAX_N,
           checksum_accum, 1, (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
