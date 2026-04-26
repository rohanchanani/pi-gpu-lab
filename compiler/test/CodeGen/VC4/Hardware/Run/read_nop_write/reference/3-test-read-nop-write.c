#include "rpi.h"
#include "read_nop_write_launch.h"

enum {
    READ_NOP_WRITE_MAX_WORDS = VC4_RUNTIME_MAX_QPUS * VC4_RUNTIME_LANE_WIDTH
};

static uint32_t input_words[READ_NOP_WRITE_MAX_WORDS];
static uint32_t result_words[READ_NOP_WRITE_MAX_WORDS];
static uint32_t expected_words[READ_NOP_WRITE_MAX_WORDS];

static void fill_inputs(uint32_t words)
{
    for (uint32_t i = 0; i < words; ++i)
    {
        input_words[i] = i + 1;
        result_words[i] = 0xdead0000u | i;
        expected_words[i] = input_words[i];
    }
}

static uint32_t checksum_words(const uint32_t *values, uint32_t words)
{
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < words; ++i)
        checksum += values[i];
    return checksum;
}

static void verify_results(
    uint32_t words,
    uint32_t *mismatches,
    uint32_t *checksum)
{
    *mismatches = 0;
    *checksum = checksum_words(result_words, words);

    for (uint32_t i = 0; i < words; ++i)
    {
        if (result_words[i] != expected_words[i])
        {
            if (*mismatches < 8)
            {
                printk("ERROR: i=%u got=%u expected=%u input=%u\n",
                       i, result_words[i], expected_words[i], input_words[i]);
            }
            (*mismatches)++;
        }
    }
}

void notmain(void)
{
    struct vc4_runtime rt;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    uint32_t active_qpus = vc4_runtime_active_qpus(&rt);
    uint32_t lane_width = vc4_runtime_lane_width();
    uint32_t words = active_qpus * lane_width;

    if (words > READ_NOP_WRITE_MAX_WORDS)
        panic("read_nop_write words=%u exceeds max=%u",
              words, READ_NOP_WRITE_MAX_WORDS);

    fill_inputs(words);

    printk("Running VC4 read_nop_write reference bundle...\n");
    int start_time = timer_get_usec();
    if (read_nop_write_launch(&rt, input_words, result_words, words) < 0)
        panic("read_nop_write launch failed");
    int end_time = timer_get_usec();

    uint32_t mismatches = 0;
    uint32_t checksum = 0;
    verify_results(words, &mismatches, &checksum);

    printk("read_nop_write qpus=%u lanes=%u words=%u checksum=%u\n",
           active_qpus, lane_width, words, checksum);

    for (uint32_t i = 0; i < 4 && i < words; ++i)
    {
        printk("sample[%u] input=%u result=%u expected=%u\n",
               i, input_words[i], result_words[i], expected_words[i]);
    }

    if (mismatches)
        panic("read_nop_write verification failed: mismatches=%u checksum=%u",
              mismatches, checksum);

    printk("VC4_TEST_RESULT name=read_nop_write status=PASS mismatches=%u "
           "active_qpus=%u words=%u checksum=%u elapsed_usec=%d\n",
           mismatches, active_qpus, words, checksum, end_time - start_time);

    printk("SUCCESS: read_nop_write verified.\n");
    vc4_runtime_shutdown(&rt);
}
