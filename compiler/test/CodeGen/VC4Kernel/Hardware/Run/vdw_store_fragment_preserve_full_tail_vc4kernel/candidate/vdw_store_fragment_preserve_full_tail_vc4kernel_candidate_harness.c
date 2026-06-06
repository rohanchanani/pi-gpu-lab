#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_LANES 16u
#define VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_ACTIVE_QPUS 1u
#define VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_BUFFER_N 64u
#define VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_TAG_FULL 0x78000000u
#define VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_TAG_EMPTY 0x79000000u
#define VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_TAG_TAIL 0x7a000000u

enum preserve_case_kind {
    PRESERVE_CASE_FULL,
    PRESERVE_CASE_EMPTY,
    PRESERVE_CASE_TAIL,
};

struct preserve_case {
    enum preserve_case_kind kind;
    uint32_t offset_elems;
    uint32_t n;
    uint32_t base_value;
};

static const struct preserve_case cases[] = {
    {PRESERVE_CASE_FULL, 0u, 16u, 0u},
    {PRESERVE_CASE_FULL, 16u, 16u, 0x100u},
    {PRESERVE_CASE_EMPTY, 32u, 0u, 0x180u},
    {PRESERVE_CASE_TAIL, 32u, 0u, 0x200u},
    {PRESERVE_CASE_TAIL, 32u, 1u, 0x200u},
    {PRESERVE_CASE_TAIL, 32u, 15u, 0x200u},
    {PRESERVE_CASE_TAIL, 32u, 16u, 0x200u},
};

static uint32_t out_values[VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_BUFFER_N];

static uint32_t sentinel_value(uint32_t index) {
    return 0xdead0000u + index;
}

static uint32_t tag_for_kind(enum preserve_case_kind kind) {
    if (kind == PRESERVE_CASE_FULL)
        return VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_TAG_FULL;
    if (kind == PRESERVE_CASE_EMPTY)
        return VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_TAG_EMPTY;
    return VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_TAG_TAIL;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_BUFFER_N; i++)
        out_values[i] = sentinel_value(i);
}

static uint32_t active_count(const struct preserve_case *tc) {
    if (tc->kind == PRESERVE_CASE_EMPTY)
        return 0u;
    if (tc->kind == PRESERVE_CASE_FULL)
        return VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_LANES;
    return tc->n;
}

static uint32_t expected_active_value(const struct preserve_case *tc, uint32_t lane) {
    return tag_for_kind(tc->kind) + tc->base_value + lane;
}

static int verify_case(const struct preserve_case *tc, int *sentinel_mismatches_out) {
    int mismatches = 0;
    int sentinel_mismatches = 0;
    uint32_t count = active_count(tc);
    for (uint32_t i = 0; i < VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_BUFFER_N; i++) {
        int active = (i >= tc->offset_elems && i < tc->offset_elems + count);
        uint32_t expected = active ? expected_active_value(tc, i - tc->offset_elems)
                                   : sentinel_value(i);
        if (out_values[i] != expected) {
            if (active) {
                if (mismatches < 8)
                    printk("ERROR: vdw_preserve_full_tail active kind=%d offset=%d n=%d i=%d gpu=%x expected=%x\n",
                           (int)tc->kind, (int)tc->offset_elems, (int)tc->n,
                           (int)i, out_values[i], expected);
                mismatches++;
            } else {
                if (sentinel_mismatches < 8)
                    printk("ERROR: vdw_preserve_full_tail sentinel kind=%d offset=%d n=%d i=%d gpu=%x expected=%x\n",
                           (int)tc->kind, (int)tc->offset_elems, (int)tc->n,
                           (int)i, out_values[i], expected);
                sentinel_mismatches++;
            }
        }
    }
    *sentinel_mismatches_out = sentinel_mismatches;
    return mismatches;
}

static int checksum_low16(const struct preserve_case *tc) {
    int checksum = 0;
    uint32_t count = active_count(tc);
    for (uint32_t lane = 0; lane < count; lane++)
        checksum += (int)(out_values[tc->offset_elems + lane] & 0xffffu);
    return checksum;
}

static int launch_case(struct vc4_program *program, vc4_deviceptr_t out_dev,
                       const struct preserve_case *tc) {
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_LANES, 1, 1);
    if (tc->kind == PRESERVE_CASE_FULL)
        return vdw_store_fragment_preserve_full_vc4kernel_launch(
            program, grid, block, out_dev, tc->offset_elems, tc->base_value);
    if (tc->kind == PRESERVE_CASE_EMPTY)
        return vdw_store_fragment_preserve_empty_vc4kernel_launch(
            program, grid, block, out_dev, tc->offset_elems, tc->base_value);
    return vdw_store_fragment_preserve_tail_vc4kernel_launch(
        program, grid, block, out_dev, tc->offset_elems, tc->n, tc->base_value);
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("vdw_store_fragment_preserve_full_tail_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_full = 0;
    int saw_empty = 0;
    int saw_tail0 = 0;
    int saw_tail15 = 0;
    int saw_tail16 = 0;
    int start = timer_get_usec();

    printk("Running VC4 vdw_store_fragment_preserve_full_tail_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct preserve_case *tc = &cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            launch_case(program, out_dev, tc) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: vdw_store_fragment_preserve_full_tail launch/copy failed case=%d kind=%d n=%d\n",
                   (int)case_id, (int)tc->kind, (int)tc->n);
            launch_failures++;
            continue;
        }

        int sentinels = 0;
        int mismatches = verify_case(tc, &sentinels);
        int checksum = checksum_low16(tc);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (tc->kind == PRESERVE_CASE_FULL)
            saw_full = 1;
        if (tc->kind == PRESERVE_CASE_EMPTY)
            saw_empty = 1;
        if (tc->kind == PRESERVE_CASE_TAIL && tc->n == 0u)
            saw_tail0 = 1;
        if (tc->kind == PRESERVE_CASE_TAIL && tc->n == 15u)
            saw_tail15 = 1;
        if (tc->kind == PRESERVE_CASE_TAIL && tc->n == 16u)
            saw_tail16 = 1;
        printk("VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_CASE case=%d kind=%d offset=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)tc->kind, (int)tc->offset_elems, (int)tc->n,
               mismatches, sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_full && saw_empty &&
                          saw_tail0 && saw_tail15 && saw_tail16) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=vdw_store_fragment_preserve_full_tail_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d buffer_n=%d saw_full=%d saw_empty=%d saw_tail0=%d saw_tail15=%d saw_tail16=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_ACTIVE_QPUS,
           VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_LANES,
           VDW_STORE_FRAGMENT_PRESERVE_FULL_TAIL_BUFFER_N,
           saw_full, saw_empty, saw_tail0, saw_tail15, saw_tail16,
           checksum_accum, 1, (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
