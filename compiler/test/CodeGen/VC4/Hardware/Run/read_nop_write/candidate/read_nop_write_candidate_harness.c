#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define READ_NOP_WRITE_ACTIVE_QPUS 12u
#define READ_NOP_WRITE_LANE_WIDTH 16u
#define READ_NOP_WRITE_WORDS (READ_NOP_WRITE_ACTIVE_QPUS * READ_NOP_WRITE_LANE_WIDTH)

static uint32_t input_words[READ_NOP_WRITE_WORDS];
static uint32_t result_words[READ_NOP_WRITE_WORDS];
static uint32_t expected_words[READ_NOP_WRITE_WORDS];

static void fill_inputs(uint32_t words) {
    for (uint32_t i = 0; i < words; ++i) {
        input_words[i] = i + 1;
        result_words[i] = 0xdead0000u | i;
        expected_words[i] = input_words[i];
    }
}

static uint32_t checksum_words(const uint32_t *values, uint32_t words) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < words; ++i)
        checksum += values[i];
    return checksum;
}

static void verify_results(uint32_t words, uint32_t *mismatches, uint32_t *checksum) {
    *mismatches = 0;
    *checksum = checksum_words(result_words, words);
    for (uint32_t i = 0; i < words; ++i) {
        if (result_words[i] != expected_words[i]) {
            if (*mismatches < 8)
                printk("ERROR: i=%u got=%u expected=%u input=%u\n", i, result_words[i], expected_words[i], input_words[i]);
            (*mismatches)++;
        }
    }
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    fill_inputs(READ_NOP_WRITE_WORDS);
    vc4_deviceptr_t input_dev = 0, result_dev = 0;
    uint32_t bytes = READ_NOP_WRITE_WORDS * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &input_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &result_dev, bytes) < 0 ||
        vc4_m2_copy_htod(program, input_dev, input_words, bytes) < 0 ||
        vc4_m2_copy_htod(program, result_dev, result_words, bytes) < 0)
        panic("read_nop_write device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(READ_NOP_WRITE_WORDS, 1, 1);

    printk("Running VC4 read_nop_write M2 candidate bundle...\n");
    int start_time = timer_get_usec();
    int launch_failures = 0;
    if (read_nop_write_launch(program, grid, block, input_dev, result_dev, READ_NOP_WRITE_WORDS) < 0 ||
        vc4_m2_copy_dtoh(program, result_words, result_dev, bytes) < 0)
        launch_failures++;
    int elapsed_usec = timer_get_usec() - start_time;

    uint32_t mismatches = 0, checksum = 0;
    verify_results(READ_NOP_WRITE_WORDS, &mismatches, &checksum);

    printk("VC4_TEST_RESULT name=read_nop_write status=%s mismatches=%u active_qpus=%u words=%u checksum=%u elapsed_usec=%d\n",
           (mismatches || launch_failures) ? "FAIL" : "PASS", mismatches,
           READ_NOP_WRITE_ACTIVE_QPUS, READ_NOP_WRITE_WORDS, checksum, elapsed_usec);
    if (mismatches)
        panic("read_nop_write verification failed: mismatches=%u checksum=%u", mismatches, checksum);

    vc4Free(program, input_dev);
    vc4Free(program, result_dev);
    vc4_program_destroy(program);
}
