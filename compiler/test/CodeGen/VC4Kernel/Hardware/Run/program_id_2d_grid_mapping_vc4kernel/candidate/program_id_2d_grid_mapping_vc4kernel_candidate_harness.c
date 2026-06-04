#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_LANES 16u
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_ACCEPTED_BLOCK_X 16u
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_OLD_BLOCK_X 192u
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_MAX_GRID_X 2u
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_MAX_GRID_Y 3u
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_MAX_REQUESTS \
    (PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_MAX_GRID_X * PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_MAX_GRID_Y)
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_ACTIVE_N \
    (PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_MAX_REQUESTS * PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_LANES)
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_GUARD 32u
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_BUFFER_N \
    (PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_ACTIVE_N + PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_GUARD)
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_SENTINEL 0xdeadbeefu
#define PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_TAG 0x72000000u

struct case_spec {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t block_x;
    uint32_t diagnostic_only;
};

static const struct case_spec cases[] = {
    {1u, 1u, PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_ACCEPTED_BLOCK_X, 0u},
    {2u, 3u, PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_ACCEPTED_BLOCK_X, 0u},
    {2u, 3u, PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_OLD_BLOCK_X, 1u},
};

static uint32_t out_values[PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_SENTINEL;
}

static uint32_t compact_request(uint32_t grid_x, uint32_t pid_x, uint32_t pid_y) {
    return pid_y * grid_x + pid_x;
}

static uint32_t expected_value(uint32_t grid_x, uint32_t grid_y,
                               uint32_t pid_x, uint32_t pid_y, uint32_t lane) {
    uint32_t request = compact_request(grid_x, pid_x, pid_y);
    return PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_TAG +
           pid_x +
           10u * pid_y +
           100u * grid_x +
           1000u * grid_y +
           10000u * request +
           lane;
}

static uint32_t runtime_request_count(const struct case_spec *spec) {
    uint32_t logical_elements = spec->grid_x * spec->grid_y * spec->block_x;
    return (logical_elements + PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_LANES - 1u) /
           PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_LANES;
}

static uint32_t gemm_expected_requests(const struct case_spec *spec) {
    return spec->grid_x * spec->grid_y;
}

static int verify_active(const struct case_spec *spec) {
    int mismatches = 0;
    for (uint32_t pid_y = 0; pid_y < spec->grid_y; pid_y++) {
        for (uint32_t pid_x = 0; pid_x < spec->grid_x; pid_x++) {
            uint32_t request = compact_request(spec->grid_x, pid_x, pid_y);
            for (uint32_t lane = 0; lane < PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_LANES; lane++) {
                uint32_t index = request * PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_LANES + lane;
                uint32_t expected = expected_value(spec->grid_x, spec->grid_y, pid_x, pid_y, lane);
                if (out_values[index] != expected) {
                    if (mismatches < 8)
                        printk("ERROR: program_id_2d active grid=(%d,%d) block_x=%d pid=(%d,%d) lane=%d gpu=%x expected=%x\n",
                               (int)spec->grid_x, (int)spec->grid_y, (int)spec->block_x,
                               (int)pid_x, (int)pid_y, (int)lane, out_values[index], expected);
                    mismatches++;
                }
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct case_spec *spec) {
    int mismatches = 0;
    uint32_t active_n = gemm_expected_requests(spec) *
                        PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_LANES;
    for (uint32_t i = active_n; i < PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: program_id_2d sentinel grid=(%d,%d) block_x=%d index=%d gpu=%x expected=%x\n",
                       (int)spec->grid_x, (int)spec->grid_y, (int)spec->block_x,
                       (int)i, out_values[i], PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int compact_records_present(const struct case_spec *spec) {
    for (uint32_t pid_y = 0; pid_y < spec->grid_y; pid_y++) {
        for (uint32_t pid_x = 0; pid_x < spec->grid_x; pid_x++) {
            uint32_t request = compact_request(spec->grid_x, pid_x, pid_y);
            uint32_t base = request * PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_LANES;
            if (out_values[base] != expected_value(spec->grid_x, spec->grid_y, pid_x, pid_y, 0u))
                return 0;
        }
    }
    return 1;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("program_id_2d_grid_mapping_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int natural_cases = 0;
    int diagnostic_cases = 0;
    int old_suspect_runtime_requests = 0;
    int old_suspect_expected_requests = 0;
    int old_suspect_overlaunch_factor = 0;
    int old_suspect_compact_records = 0;
    int start = timer_get_usec();

    printk("Running VC4 program_id_2d_grid_mapping_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_spec *spec = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(spec->grid_x, spec->grid_y, 1);
        vc4_dim3 block = vc4_m2_dim3(spec->block_x, 1, 1);
        uint32_t runtime_requests = runtime_request_count(spec);
        uint32_t expected_requests = gemm_expected_requests(spec);

        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            program_id_2d_grid_mapping_vc4kernel_launch(program, grid, block, out_dev) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: program_id_2d_grid_mapping_vc4kernel launch/copy failed case=%d grid=(%d,%d) block_x=%d\n",
                   (int)case_id, (int)spec->grid_x, (int)spec->grid_y, (int)spec->block_x);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(spec);
        int sentinels = verify_sentinels(spec);
        int compact = compact_records_present(spec);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        if (spec->diagnostic_only) {
            diagnostic_cases++;
            old_suspect_runtime_requests = (int)runtime_requests;
            old_suspect_expected_requests = (int)expected_requests;
            old_suspect_overlaunch_factor =
                expected_requests == 0u ? 0 : (int)(runtime_requests / expected_requests);
            old_suspect_compact_records = compact;
        } else {
            natural_cases++;
        }

        printk("PROGRAM_ID_2D_GRID_MAPPING_CASE case=%d grid_x=%d grid_y=%d block_x=%d diagnostic_only=%d expected_requests=%d runtime_requests=%d compact_records=%d mismatches=%d sentinel_mismatches=%d\n",
               (int)case_id, (int)spec->grid_x, (int)spec->grid_y, (int)spec->block_x,
               (int)spec->diagnostic_only, (int)expected_requests, (int)runtime_requests,
               compact, mismatches, sentinels);
    }

    /*
     * VC4Kernel independent-vector launches count requests as
     * ceil(grid.x * grid.y * grid.z * block.x * block.y * block.z / 16).
     * For GEMM-style 2-D row/tile kernels, use block={16,1,1}; grid then
     * carries the logical row/tile request space.  block.x=12*16 is a
     * diagnostic overlaunch shape, not the accepted GEMM row/tile convention.
     */
    int elapsed = timer_get_usec() - start;
    int old_suspect_diagnostic_only = diagnostic_cases == 1;
    int gemm_fixture_launch_geometry_bug =
        old_suspect_runtime_requests == 72 &&
        old_suspect_expected_requests == 6 &&
        old_suspect_overlaunch_factor == 12 &&
        old_suspect_compact_records;
    const char *status = (total_mismatches == 0 &&
                          sentinel_mismatches == 0 &&
                          launch_failures == 0 &&
                          natural_cases == 2 &&
                          old_suspect_diagnostic_only &&
                          gemm_fixture_launch_geometry_bug) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=program_id_2d_grid_mapping_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d natural_cases=%d diagnostic_cases=%d accepted_block_x=%d natural_1x1_requests=1 natural_2x3_requests=6 old_suspect_block_x=%d old_suspect_runtime_requests=%d old_suspect_gemm_expected_requests=%d old_suspect_overlaunch_factor=%d old_suspect_diagnostic_only=%d gemm_fixture_launch_geometry_bug=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           natural_cases, diagnostic_cases,
           PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_ACCEPTED_BLOCK_X,
           PROGRAM_ID_2D_GRID_MAPPING_VC4KERNEL_OLD_BLOCK_X,
           old_suspect_runtime_requests,
           old_suspect_expected_requests,
           old_suspect_overlaunch_factor,
           old_suspect_diagnostic_only,
           gemm_fixture_launch_geometry_bug,
           (int)program_id_2d_grid_mapping_vc4kernel_runtime_allocations(),
           (int)program_id_2d_grid_mapping_vc4kernel_runtime_launches(),
           elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
