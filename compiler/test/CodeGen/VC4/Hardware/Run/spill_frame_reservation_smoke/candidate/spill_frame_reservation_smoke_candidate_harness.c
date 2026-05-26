#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS 12u
#define SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH 16u
#define SPILL_FRAME_RESERVATION_SMOKE_WORDS \
    (SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS * SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH)
#define SPILL_FRAME_RESERVATION_SMOKE_GUARD_WORDS 16u
#define SPILL_FRAME_RESERVATION_SMOKE_SPILL_FRAME_BYTES 20u
#define SPILL_FRAME_RESERVATION_SMOKE_SPILL_FRAME_STRIDE_BYTES 64u
#define SPILL_FRAME_RESERVATION_SMOKE_SPILL_FRAME_COUNT SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS
#define SPILL_FRAME_RESERVATION_SMOKE_SPILL_ARENA_BYTES \
    (SPILL_FRAME_RESERVATION_SMOKE_SPILL_FRAME_STRIDE_BYTES * \
     SPILL_FRAME_RESERVATION_SMOKE_SPILL_FRAME_COUNT)
#define SPILL_FRAME_RESERVATION_SMOKE_SENTINEL_BASE 0xdead0000u

static uint32_t gpu_result_words[SPILL_FRAME_RESERVATION_SMOKE_WORDS +
                                 SPILL_FRAME_RESERVATION_SMOKE_GUARD_WORDS];

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

static void fill_expected_guard(void) {
    for (uint32_t i = 0; i < SPILL_FRAME_RESERVATION_SMOKE_GUARD_WORDS; i++)
        gpu_result_words[SPILL_FRAME_RESERVATION_SMOKE_WORDS + i] =
            SPILL_FRAME_RESERVATION_SMOKE_SENTINEL_BASE + i;
}

static int verify_guard(uint32_t *first_bad_index, uint32_t *first_bad_value,
                        uint32_t *first_bad_expected) {
    int mismatches = 0;
    *first_bad_index = 0;
    *first_bad_value = 0;
    *first_bad_expected = 0;
    for (uint32_t i = 0; i < SPILL_FRAME_RESERVATION_SMOKE_GUARD_WORDS; i++) {
        uint32_t index = SPILL_FRAME_RESERVATION_SMOKE_WORDS + i;
        uint32_t expected = SPILL_FRAME_RESERVATION_SMOKE_SENTINEL_BASE + i;
        uint32_t actual = gpu_result_words[index];
        if (actual != expected) {
            if (mismatches == 0) {
                *first_bad_index = index;
                *first_bad_value = actual;
                *first_bad_expected = expected;
            }
            if (mismatches < 8)
                printk("ERROR: guard index=%d actual=0x%x expected=0x%x\n",
                       index, actual, expected);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    fill_expected_guard();
    uint32_t bytes = (SPILL_FRAME_RESERVATION_SMOKE_WORDS +
                      SPILL_FRAME_RESERVATION_SMOKE_GUARD_WORDS) *
                     sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, gpu_result_words, bytes) < 0)
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
    uint32_t first_bad_guard_index = 0, first_bad_guard_value = 0, first_bad_guard_expected = 0;
    int sentinel_mismatches =
        verify_guard(&first_bad_guard_index, &first_bad_guard_value, &first_bad_guard_expected);
    uint32_t checksum = checksum_words(gpu_result_words, SPILL_FRAME_RESERVATION_SMOKE_WORDS);
    uint32_t runtime_allocations = spill_frame_reservation_smoke_runtime_allocations();
    uint32_t runtime_launches = spill_frame_reservation_smoke_runtime_launches();
    uint32_t runtime_capacity = spill_frame_reservation_smoke_runtime_capacity();
    uint32_t code_uploads = spill_frame_reservation_smoke_runtime_code_uploads();
    uint32_t recorded_launch_failures = spill_frame_reservation_smoke_runtime_launch_failures();

    printk("spill_frame_reservation_smoke qpus=%d lanes=%d words=%d checksum=%d spill_arena_bytes=%d\n",
           (int)SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS,
           (int)SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH,
           (int)SPILL_FRAME_RESERVATION_SMOKE_WORDS, (int)checksum,
           (int)SPILL_FRAME_RESERVATION_SMOKE_SPILL_ARENA_BYTES);
    printk("VC4_TEST_RESULT name=spill_frame_reservation_smoke status=%s checked_elements=%d mismatches=%d sentinel_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d words=%d checksum=%d spill_frame_bytes=%d spill_frame_stride_bytes=%d spill_frame_count=%d hidden_spill_arena_bytes=%d runtime_allocations=%d runtime_launches=%d runtime_capacity=%d code_uploads=%d elapsed_usec=%d\n",
           (mismatches || sentinel_mismatches || launch_failures || recorded_launch_failures ||
            runtime_allocations != 1u || runtime_launches != 1u || code_uploads != 1u)
               ? "FAIL"
               : "PASS",
           (int)SPILL_FRAME_RESERVATION_SMOKE_WORDS, mismatches, sentinel_mismatches,
           launch_failures, (int)recorded_launch_failures,
           (int)SPILL_FRAME_RESERVATION_SMOKE_ACTIVE_QPUS,
           (int)SPILL_FRAME_RESERVATION_SMOKE_LANE_WIDTH,
           (int)SPILL_FRAME_RESERVATION_SMOKE_WORDS, (int)checksum,
           (int)SPILL_FRAME_RESERVATION_SMOKE_SPILL_FRAME_BYTES,
           (int)SPILL_FRAME_RESERVATION_SMOKE_SPILL_FRAME_STRIDE_BYTES,
           (int)SPILL_FRAME_RESERVATION_SMOKE_SPILL_FRAME_COUNT,
           (int)SPILL_FRAME_RESERVATION_SMOKE_SPILL_ARENA_BYTES,
           (int)runtime_allocations, (int)runtime_launches,
           (int)runtime_capacity, (int)code_uploads, elapsed_usec);
    if (mismatches)
        panic("spill_frame_reservation_smoke verification failed: first_bad_index=%d actual=%d expected=%d",
              first_bad_index, first_bad_value, first_bad_expected);
    if (sentinel_mismatches)
        panic("spill_frame_reservation_smoke guard failed: first_bad_index=%d actual=0x%x expected=0x%x",
              first_bad_guard_index, first_bad_guard_value, first_bad_guard_expected);
    if (launch_failures || recorded_launch_failures)
        panic("spill_frame_reservation_smoke launch failed: launch_failures=%d recorded=%d",
              launch_failures, recorded_launch_failures);
    if (runtime_allocations != 1u || runtime_launches != 1u || code_uploads != 1u)
        panic("spill_frame_reservation_smoke runtime counters failed: allocations=%d launches=%d code_uploads=%d",
              runtime_allocations, runtime_launches, code_uploads);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
