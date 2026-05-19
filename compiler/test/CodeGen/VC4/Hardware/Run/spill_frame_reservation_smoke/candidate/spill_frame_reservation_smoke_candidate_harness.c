#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS 12u
#define SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH 16u
#define SPILL_FRAME_RESERVATION_SMOKE_WORDS \
    (SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS * SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH)

static uint32_t gpu_result_words[SPILL_FRAME_RESERVATION_SMOKE_WORDS];

static uint32_t checksum_words(const uint32_t *values, uint32_t words) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < words; i++)
        checksum += values[i];
    return checksum;
}

static int verify_output(uint32_t *first_bad_index, uint32_t *first_bad_value, uint32_t *first_bad_expected) {
    int mismatches = 0;
    *first_bad_index = 0;
    *first_bad_value = 0;
    *first_bad_expected = 0;
    for (uint32_t qpu = 0; qpu < SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS; qpu++) {
        for (uint32_t lane = 0; lane < SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH; lane++) {
            uint32_t index = qpu * SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH + lane;
            uint32_t expected = qpu;
            uint32_t actual = gpu_result_words[index];
            if (actual != expected) {
                if (mismatches == 0) {
                    *first_bad_index = index;
                    *first_bad_value = actual;
                    *first_bad_expected = expected;
                }
                if (mismatches < 8)
                    printk("ERROR: index=%d actual=%d expected=%d\n", index, actual, expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = SPILL_FRAME_RESERVATION_SMOKE_WORDS * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0 ||
        vc4MemsetD8(program, out_dev, 0, bytes) < 0)
        panic("spill_frame_reservation_smoke device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(SPILL_FRAME_RESERVATION_SMOKE_WORDS, 1, 1);

    printk("Running VC4 spill_frame_reservation_smoke M2 candidate bundle...\n");
    int start_time = timer_get_usec();
    int launch_failures = 0;
    if (spill_frame_reservation_smoke_launch(program, grid, block, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, gpu_result_words, out_dev, bytes) < 0)
        launch_failures++;
    int elapsed_usec = timer_get_usec() - start_time;

    uint32_t first_bad_index = 0, first_bad_value = 0, first_bad_expected = 0;
    int mismatches = verify_output(&first_bad_index, &first_bad_value, &first_bad_expected);
    uint32_t checksum = checksum_words(gpu_result_words, SPILL_FRAME_RESERVATION_SMOKE_WORDS);

    printk("spill_frame_reservation_smoke qpus=%d lanes=%d words=%d checksum=%d\n",
           (int)SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS,
           (int)SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH,
           (int)SPILL_FRAME_RESERVATION_SMOKE_WORDS, (int)checksum);
    printk("VC4_TEST_RESULT name=spill_frame_reservation_smoke status=%s mismatches=%d active_qpus=%d words=%d checksum=%d elapsed_usec=%d\n",
           (mismatches || launch_failures) ? "FAIL" : "PASS", mismatches,
           (int)SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS,
           (int)SPILL_FRAME_RESERVATION_SMOKE_WORDS, (int)checksum, elapsed_usec);
    if (mismatches)
        panic("spill_frame_reservation_smoke verification failed: first_bad_index=%d actual=%d expected=%d",
              first_bad_index, first_bad_value, first_bad_expected);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
