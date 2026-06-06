#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 5u
#define OUT_WORDS (LANES * SEGMENTS + 16u)
#define SENTINEL 0xdeadbeefu
#define TAG 0x78000000u

struct rect_case {
    uint32_t row;
    uint32_t rows;
    uint32_t col_base;
    uint32_t cols;
};

static const struct rect_case cases[SEGMENTS] = {
    {0u, 1u, 0u, 16u},
    {0u, 1u, 3u, 8u},
    {1u, 1u, 3u, 8u},
    {0u, 1u, 7u, 2u},
    {0u, 1u, 15u, 1u},
};

static uint32_t input_values[LANES];
static uint32_t out_values[OUT_WORDS];

static void fill_buffers(void) {
    for (uint32_t i = 0; i < LANES; i++)
        input_values[i] = TAG + i;
    for (uint32_t i = 0; i < OUT_WORDS; i++)
        out_values[i] = SENTINEL;
}

static int lane_active(const struct rect_case *tc, uint32_t lane) {
    if (tc->row >= tc->rows || tc->cols == 0u || tc->col_base >= LANES)
        return 0;
    return lane >= tc->col_base && lane < tc->col_base + tc->cols;
}

static int verify_segment(uint32_t segment, const struct rect_case *tc) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < LANES; lane++) {
        uint32_t expected = lane_active(tc, lane) ? (TAG + lane) : 0u;
        uint32_t got = out_values[segment * LANES + lane];
        if (got != expected) {
            if (mismatches < 8)
                printk("ERROR: safe_rect segment=%d row=%d rows=%d col_base=%d cols=%d lane=%d gpu=%x expected=%x\n",
                       (int)segment, (int)tc->row, (int)tc->rows,
                       (int)tc->col_base, (int)tc->cols, (int)lane, got,
                       expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES * SEGMENTS; i < OUT_WORDS; i++) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: safe_rect sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < LANES * SEGMENTS; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("tmu_load_safe_offset_rect_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_center_rect = 0;
    int saw_empty_row = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 tmu_load_safe_offset_rect_vc4kernel candidate bundle...\n");
    fill_buffers();
    if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
        tmu_load_safe_offset_rect_vc4kernel_launch(program, grid, block, input_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0) {
        printk("ERROR: safe_rect launch/copy failed\n");
        launch_failures++;
    } else {
        for (uint32_t segment = 0; segment < SEGMENTS; segment++) {
            const struct rect_case *tc = &cases[segment];
            int mismatches = verify_segment(segment, tc);
            total_mismatches += mismatches;
            saw_center_rect |= (tc->row == 0u && tc->rows == 1u &&
                                tc->col_base == 3u && tc->cols == 8u);
            saw_empty_row |= (tc->row >= tc->rows);
            printk("TMU_LOAD_SAFE_OFFSET_RECT_CASE segment=%d row=%d rows=%d col_base=%d cols=%d mismatches=%d\n",
                   (int)segment, (int)tc->row, (int)tc->rows,
                   (int)tc->col_base, (int)tc->cols, mismatches);
        }
        sentinel_mismatches = verify_sentinels();
        checksum_accum = checksum_low16();
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_center_rect &&
                          saw_empty_row) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=tmu_load_safe_offset_rect_vc4kernel status=%s cases=5 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_center_rect=%d saw_empty_row=%d poison_inactive_offsets=1 checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           LANES, saw_center_rect, saw_empty_row, checksum_accum,
           (int)tmu_load_safe_offset_rect_vc4kernel_runtime_allocations(),
           (int)tmu_load_safe_offset_rect_vc4kernel_runtime_launches(), elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
