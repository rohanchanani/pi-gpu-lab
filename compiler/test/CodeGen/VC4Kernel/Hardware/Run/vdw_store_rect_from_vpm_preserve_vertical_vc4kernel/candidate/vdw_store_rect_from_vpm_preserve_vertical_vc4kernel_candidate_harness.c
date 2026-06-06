#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define VDW_RECT_VERTICAL_INPUT_ROWS 8u
#define VDW_RECT_VERTICAL_INPUT_COLS 4u
#define VDW_RECT_VERTICAL_OUTPUT_ROWS 4u
#define VDW_RECT_VERTICAL_OUTPUT_COLS 8u
#define VDW_RECT_VERTICAL_MAX_STRIDE_WORDS 19u
#define VDW_RECT_VERTICAL_INPUT_WORDS (((VDW_RECT_VERTICAL_INPUT_ROWS - 1u) * VDW_RECT_VERTICAL_MAX_STRIDE_WORDS) + VDW_RECT_VERTICAL_INPUT_COLS)
#define VDW_RECT_VERTICAL_OUTPUT_WORDS (((VDW_RECT_VERTICAL_OUTPUT_ROWS - 1u) * VDW_RECT_VERTICAL_MAX_STRIDE_WORDS) + VDW_RECT_VERTICAL_OUTPUT_COLS)
#define VDW_RECT_VERTICAL_GUARD 32u

struct vertical_case {
    uint32_t active_rows;
    uint32_t active_cols;
    uint32_t pitch_stride_words;
};

static const struct vertical_case cases[] = {
    {0u, 8u, 16u},
    {1u, 0u, 16u},
    {1u, 1u, 19u},
    {2u, 3u, 16u},
    {4u, 8u, 19u},
};

static uint32_t input_values[VDW_RECT_VERTICAL_INPUT_WORDS + VDW_RECT_VERTICAL_GUARD];
static uint32_t output_values[VDW_RECT_VERTICAL_OUTPUT_WORDS + VDW_RECT_VERTICAL_GUARD];

static uint32_t pattern(uint32_t row, uint32_t col) {
    return 0x66000000u | (row << 8) | col;
}

static uint32_t sentinel_value(uint32_t index) {
    return 0xb6b60000u + index;
}

static void fill_buffers(const struct vertical_case *tc) {
    for (uint32_t i = 0; i < VDW_RECT_VERTICAL_INPUT_WORDS + VDW_RECT_VERTICAL_GUARD; i++)
        input_values[i] = sentinel_value(i);
    for (uint32_t row = 0; row < VDW_RECT_VERTICAL_INPUT_ROWS; row++)
        for (uint32_t col = 0; col < VDW_RECT_VERTICAL_INPUT_COLS; col++)
            input_values[row * tc->pitch_stride_words + col] = pattern(row, col);
    for (uint32_t i = 0; i < VDW_RECT_VERTICAL_OUTPUT_WORDS + VDW_RECT_VERTICAL_GUARD; i++)
        output_values[i] = sentinel_value(i);
}

static uint32_t expected_at(const struct vertical_case *tc, uint32_t index,
                            int *active_out) {
    *active_out = 0;
    for (uint32_t row = 0; row < VDW_RECT_VERTICAL_OUTPUT_ROWS; row++) {
        uint32_t base = row * tc->pitch_stride_words;
        if (index >= base && index < base + VDW_RECT_VERTICAL_OUTPUT_COLS) {
            uint32_t col = index - base;
            if (row < tc->active_rows && col < tc->active_cols) {
                *active_out = 1;
                return pattern(col, row);
            }
            return sentinel_value(index);
        }
    }
    return sentinel_value(index);
}

static int verify_case(const struct vertical_case *tc,
                       int *sentinel_mismatches_out,
                       uint32_t *checksum_out) {
    int mismatches = 0;
    int sentinel_mismatches = 0;
    *checksum_out = 0;
    for (uint32_t i = 0; i < VDW_RECT_VERTICAL_OUTPUT_WORDS + VDW_RECT_VERTICAL_GUARD; i++) {
        int active = 0;
        uint32_t expected = expected_at(tc, i, &active);
        uint32_t got = output_values[i];
        if (active)
            *checksum_out += got;
        if (got != expected) {
            if (active) {
                if (mismatches < 8)
                    printk("ERROR: vdw_rect_vertical active rows=%d cols=%d stride=%d index=%d gpu=%x expected=%x\n",
                           (int)tc->active_rows, (int)tc->active_cols,
                           (int)tc->pitch_stride_words, (int)i, got, expected);
                mismatches++;
            } else {
                if (sentinel_mismatches < 8)
                    printk("ERROR: vdw_rect_vertical sentinel rows=%d cols=%d stride=%d index=%d gpu=%x expected=%x\n",
                           (int)tc->active_rows, (int)tc->active_cols,
                           (int)tc->pitch_stride_words, (int)i, got, expected);
                sentinel_mismatches++;
            }
        }
    }
    *sentinel_mismatches_out = sentinel_mismatches;
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t in_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vdw_store_rect_from_vpm_preserve_vertical_vc4kernel program create failed");
    if (vc4_m2_malloc(program, &in_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(output_values)) < 0)
        panic("vdw_store_rect_from_vpm_preserve_vertical_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checksum_accum = 0;
    int saw_rows0 = 0;
    int saw_rows4 = 0;
    int saw_cols0 = 0;
    int saw_cols8 = 0;
    int saw_stride19 = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

    printk("Running VC4 vdw_store_rect_from_vpm_preserve_vertical_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct vertical_case *tc = &cases[case_id];
        fill_buffers(tc);
        if (vc4_m2_copy_htod(program, in_dev, input_values, sizeof(input_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, output_values, sizeof(output_values)) < 0 ||
            vdw_store_rect_from_vpm_preserve_vertical_vc4kernel_launch(
                program, grid, block, in_dev, out_dev, tc->active_rows,
                tc->active_cols, tc->pitch_stride_words * sizeof(uint32_t)) < 0 ||
            vc4_m2_copy_dtoh(program, output_values, out_dev, sizeof(output_values)) < 0) {
            printk("ERROR: vdw_rect_vertical launch/copy failed case=%d rows=%d cols=%d\n",
                   (int)case_id, (int)tc->active_rows, (int)tc->active_cols);
            launch_failures++;
            continue;
        }

        int sentinels = 0;
        uint32_t checksum = 0;
        int mismatches = verify_case(tc, &sentinels, &checksum);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (tc->active_rows == 0u)
            saw_rows0 = 1;
        if (tc->active_rows == 4u)
            saw_rows4 = 1;
        if (tc->active_cols == 0u)
            saw_cols0 = 1;
        if (tc->active_cols == 8u)
            saw_cols8 = 1;
        if (tc->pitch_stride_words == 19u)
            saw_stride19 = 1;
        printk("VDW_STORE_RECT_FROM_VPM_PRESERVE_VERTICAL_CASE case=%d rows=%d cols=%d stride=%d mismatches=%d sentinel_mismatches=%d checksum=%u\n",
               (int)case_id, (int)tc->active_rows, (int)tc->active_cols,
               (int)tc->pitch_stride_words, mismatches, sentinels, checksum);
    }

    launch_failures +=
        (int)vdw_store_rect_from_vpm_preserve_vertical_vc4kernel_runtime_launch_failures();
    uint32_t launches =
        vdw_store_rect_from_vpm_preserve_vertical_vc4kernel_runtime_launches();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_rows0 && saw_rows4 &&
                          saw_cols0 && saw_cols8 && saw_stride19)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=vdw_store_rect_from_vpm_preserve_vertical_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d max_rows=%d max_cols=%d max_stride_words=%d saw_rows0=%d saw_rows4=%d saw_cols0=%d saw_cols8=%d saw_stride19=%d checksum_accum=%u runtime_allocations=%d runtime_launches=%u elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           (int)VDW_RECT_VERTICAL_OUTPUT_ROWS,
           (int)VDW_RECT_VERTICAL_OUTPUT_COLS,
           (int)VDW_RECT_VERTICAL_MAX_STRIDE_WORDS, saw_rows0, saw_rows4,
           saw_cols0, saw_cols8, saw_stride19, checksum_accum, 2, launches,
           elapsed);

    vc4Free(program, in_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
