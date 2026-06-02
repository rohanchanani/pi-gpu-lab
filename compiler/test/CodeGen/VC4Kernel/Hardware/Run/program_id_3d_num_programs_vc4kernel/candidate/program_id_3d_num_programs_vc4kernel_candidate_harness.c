#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_ACTIVE_QPUS 12u
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES 16u
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X 3u
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y 4u
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Z 2u
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_REQUESTS \
    (PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Z)
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_ACTIVE_N \
    (PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_REQUESTS * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES)
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GUARD 32u
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_BUFFER_N \
    (PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_ACTIVE_N + PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GUARD)
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_SENTINEL 0xdeadbeefu
#define PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_TAG 0x71000000u

static uint32_t out_values[PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_SENTINEL;
}

static uint32_t linear_request(uint32_t pid_x, uint32_t pid_y, uint32_t pid_z) {
    return ((pid_z * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y + pid_y) *
            PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X) + pid_x;
}

static uint32_t expected_value(uint32_t pid_x, uint32_t pid_y, uint32_t pid_z, uint32_t lane) {
    return PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_TAG +
           pid_x +
           10u * pid_y +
           100u * pid_z +
           1000u * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X +
           10000u * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y +
           100000u * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Z +
           lane;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t pid_z = 0; pid_z < PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Z; pid_z++) {
        for (uint32_t pid_y = 0; pid_y < PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y; pid_y++) {
            for (uint32_t pid_x = 0; pid_x < PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X; pid_x++) {
                uint32_t request = linear_request(pid_x, pid_y, pid_z);
                for (uint32_t lane = 0; lane < PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES; lane++) {
                    uint32_t index = request * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES + lane;
                    uint32_t expected = expected_value(pid_x, pid_y, pid_z, lane);
                    if (out_values[index] != expected) {
                        if (mismatches < 8)
                            printk("ERROR: program_id_3d active request=%d pid=(%d,%d,%d) lane=%d gpu=%x expected=%x\n",
                                   (int)request, (int)pid_x, (int)pid_y, (int)pid_z, (int)lane,
                                   out_values[index], expected);
                        mismatches++;
                    }
                }
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_ACTIVE_N;
         i < PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: program_id_3d sentinel index=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int request_contributed(uint32_t request) {
    if (request >= PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_REQUESTS)
        return 0;
    uint32_t pid_x = request % PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X;
    uint32_t pid_y = (request / PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X) %
                     PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y;
    uint32_t pid_z = request / (PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X *
                                PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y);
    uint32_t base = request * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES;
    for (uint32_t lane = 0; lane < PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES; lane++)
        if (out_values[base + lane] != expected_value(pid_x, pid_y, pid_z, lane))
            return 0;
    return 1;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("program_id_3d_num_programs_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int start = timer_get_usec();

    vc4_dim3 grid = vc4_m2_dim3(PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X,
                                PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y,
                                PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Z);
    vc4_dim3 block = vc4_m2_dim3(PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 program_id_3d_num_programs_vc4kernel candidate bundle...\n");
    fill_output();
    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        program_id_3d_num_programs_vc4kernel_launch(program, grid, block, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
        printk("ERROR: program_id_3d_num_programs_vc4kernel launch/copy failed\n");
        launch_failures++;
    } else {
        total_mismatches += verify_active();
        sentinel_mismatches += verify_sentinels();
    }

    int saw_pid_0_0_0 =
        out_values[0] == expected_value(0u, 0u, 0u, 0u);
    int saw_pid_2_3_1 =
        out_values[linear_request(2u, 3u, 1u) * PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES] ==
        expected_value(2u, 3u, 1u, 0u);
    int saw_request_0 = request_contributed(0u);
    int saw_request_11 = request_contributed(11u);
    int saw_request_12 = request_contributed(12u);
    int saw_request_23 = request_contributed(23u);
    int waves = PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_REQUESTS /
                PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_ACTIVE_QPUS;
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0 &&
                          saw_pid_0_0_0 && saw_pid_2_3_1 &&
                          saw_request_0 && saw_request_11 && saw_request_12 && saw_request_23) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=program_id_3d_num_programs_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d grid_x=%d grid_y=%d grid_z=%d requests=%d waves=%d buffer_n=%d saw_pid_0_0_0=%d saw_pid_2_3_1=%d saw_request_0=%d saw_request_11=%d saw_request_12=%d saw_request_23=%d runtime_allocations=1 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_ACTIVE_QPUS,
           PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_LANES,
           PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_X,
           PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Y,
           PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_GRID_Z,
           PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_REQUESTS, waves,
           PROGRAM_ID_3D_NUM_PROGRAMS_VC4KERNEL_BUFFER_N,
           saw_pid_0_0_0, saw_pid_2_3_1,
           saw_request_0, saw_request_11, saw_request_12, saw_request_23, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
