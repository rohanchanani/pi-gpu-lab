#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_ROWS 3u
#define VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_COLS 16u
#define VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_WORDS (VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_ROWS * VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_COLS)
#define VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_GUARD 32u
#define VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_BASE0 0x54000000u
#define VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_BASE1 0x55000000u

static const uint32_t cases[] = {0u, 1u, 15u, 16u};
static uint32_t output_values[VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_WORDS +
                              VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_GUARD];

static uint32_t sentinel_value(uint32_t index) {
    return 0xcafe0000u + index;
}

static void fill_output(void) {
    for (uint32_t i = 0;
         i < VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_WORDS +
                 VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_GUARD;
         i++)
        output_values[i] = sentinel_value(i);
}

static uint32_t expected_value(uint32_t n, uint32_t row, uint32_t col) {
    if (row == 0u)
        return VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_BASE0 + col;
    if (row == 1u && col < n)
        return VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_BASE1 + col;
    return sentinel_value(row * VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_COLS + col);
}

static int verify_case(uint32_t n, int *sentinel_mismatches_out,
                       uint32_t *checksum_out) {
    int mismatches = 0;
    int sentinel_mismatches = 0;
    *checksum_out = 0;
    for (uint32_t row = 0; row < VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_ROWS; row++) {
        for (uint32_t col = 0; col < VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_COLS; col++) {
            uint32_t index = row * VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_COLS + col;
            uint32_t expected = expected_value(n, row, col);
            uint32_t got = output_values[index];
            int active = (row == 0u) || (row == 1u && col < n);
            if (active)
                *checksum_out += got;
            if (got != expected) {
                if (active) {
                    if (mismatches < 8)
                        printk("ERROR: vdw_vpm_tail active n=%d row=%d col=%d gpu=%x expected=%x\n",
                               (int)n, (int)row, (int)col, got, expected);
                    mismatches++;
                } else {
                    if (sentinel_mismatches < 8)
                        printk("ERROR: vdw_vpm_tail sentinel n=%d row=%d col=%d gpu=%x expected=%x\n",
                               (int)n, (int)row, (int)col, got, expected);
                    sentinel_mismatches++;
                }
            }
        }
    }
    for (uint32_t i = VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_WORDS;
         i < VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_WORDS +
                 VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_GUARD;
         i++) {
        uint32_t expected = sentinel_value(i);
        if (output_values[i] != expected) {
            if (sentinel_mismatches < 8)
                printk("ERROR: vdw_vpm_tail guard n=%d index=%d gpu=%x expected=%x\n",
                       (int)n, (int)i, output_values[i], expected);
            sentinel_mismatches++;
        }
    }
    *sentinel_mismatches_out = sentinel_mismatches;
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vdw_store_vpm_fragment_preserve_tail_vc4kernel program create failed");

    uint32_t bytes = sizeof(output_values);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("vdw_store_vpm_fragment_preserve_tail_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checksum_accum = 0;
    int saw_n0 = 0;
    int saw_n1 = 0;
    int saw_n15 = 0;
    int saw_n16 = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

    printk("Running VC4 vdw_store_vpm_fragment_preserve_tail_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t n = cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, output_values, bytes) < 0 ||
            vdw_store_vpm_fragment_preserve_tail_vc4kernel_launch(
                program, grid, block, out_dev, n) < 0 ||
            vc4_m2_copy_dtoh(program, output_values, out_dev, bytes) < 0) {
            printk("ERROR: vdw_store_vpm_fragment_preserve_tail launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }

        int sentinels = 0;
        uint32_t checksum = 0;
        int mismatches = verify_case(n, &sentinels, &checksum);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (n == 0u)
            saw_n0 = 1;
        if (n == 1u)
            saw_n1 = 1;
        if (n == 15u)
            saw_n15 = 1;
        if (n == 16u)
            saw_n16 = 1;
        printk("VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_CASE case=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%u\n",
               (int)case_id, (int)n, mismatches, sentinels, checksum);
    }

    launch_failures +=
        (int)vdw_store_vpm_fragment_preserve_tail_vc4kernel_runtime_launch_failures();
    uint32_t launches = vdw_store_vpm_fragment_preserve_tail_vc4kernel_runtime_launches();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_n0 && saw_n1 &&
                          saw_n15 && saw_n16) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=vdw_store_vpm_fragment_preserve_tail_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d rows=%d cols=%d saw_n0=%d saw_n1=%d saw_n15=%d saw_n16=%d checksum_accum=%u runtime_allocations=%d runtime_launches=%u elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           (int)VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_ROWS,
           (int)VDW_STORE_VPM_FRAGMENT_PRESERVE_TAIL_COLS,
           saw_n0, saw_n1, saw_n15, saw_n16, checksum_accum, 1, launches,
           elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
