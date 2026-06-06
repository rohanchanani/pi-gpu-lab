#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_ACTIVE_QPUS 12u
#define VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_LANES 16u
#define VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_ELEMENTS_PER_WAVE (VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_ACTIVE_QPUS * VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_LANES)
#define VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_MAX_N 33u
#define VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_MAX_COVERAGE_N 192u
#define VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_GUARD 32u
#define VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_BUFFER_N (VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_MAX_COVERAGE_N + VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_GUARD)
#define VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_TAG 0x7b000000u

static const uint32_t cases[] = {
    0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u
};

static uint32_t out_values[VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_BUFFER_N];

static uint32_t sentinel_value(uint32_t index) {
    return 0xbeef0000u + index;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_BUFFER_N; i++)
        out_values[i] = sentinel_value(i);
}

static uint32_t expected_value(uint32_t index) {
    return VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_TAG + index;
}

static int verify_active(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t expected = expected_value(i);
        if (out_values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: vdw_preserve_runtime_tail active index=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n, uint32_t coverage_n) {
    int mismatches = 0;
    for (uint32_t i = n; i < VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_BUFFER_N; i++) {
        uint32_t expected = sentinel_value(i);
        if (out_values[i] != expected) {
            if (mismatches < 8)
                printk("ERROR: vdw_preserve_runtime_tail sentinel index=%d coverage_n=%d gpu=%x expected=%x\n",
                       (int)i, (int)coverage_n, out_values[i], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_ELEMENTS_PER_WAVE - 1u) /
                     VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("vdw_store_fragment_preserve_runtime_tail_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_n0 = 0;
    int saw_n17 = 0;
    int saw_n33 = 0;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 vdw_store_fragment_preserve_runtime_tail_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t n = cases[case_id];
        uint32_t waves = rounded_waves(n);
        uint32_t coverage_n = waves * VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_ELEMENTS_PER_WAVE;
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            vdw_store_fragment_preserve_runtime_tail_vc4kernel_launch(
                program, grid, block, out_dev, n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: vdw_store_fragment_preserve_runtime_tail launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(n);
        int sentinels = verify_sentinels(n, coverage_n);
        int checksum = checksum_low16(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (n == 0u)
            saw_n0 = 1;
        if (n == 17u)
            saw_n17 = 1;
        if (n == 33u)
            saw_n33 = 1;
        printk("VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_CASE case=%d n=%d waves=%d coverage_n=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)n, (int)waves, (int)coverage_n, mismatches,
               sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_n0 && saw_n17 &&
                          saw_n33) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=vdw_store_fragment_preserve_runtime_tail_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d saw_n0=%d saw_n17=%d saw_n33=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_ACTIVE_QPUS,
           VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_LANES,
           VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_MAX_N,
           VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_MAX_COVERAGE_N,
           VDW_STORE_FRAGMENT_PRESERVE_RUNTIME_TAIL_BUFFER_N,
           saw_n0, saw_n17, saw_n33, checksum_accum,
           1, (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
