#include "rpi.h"
#include <stdint.h>
#include "tmu_read_nop_write_launch.h"

static uint32_t input_words[TMU_READ_NOP_WRITE_MAX_WORDS];
static uint32_t result_words[TMU_READ_NOP_WRITE_MAX_WORDS];
static uint32_t expected_words[TMU_READ_NOP_WRITE_MAX_WORDS];

static void fill_inputs(uint32_t words)
{
    for (uint32_t i = 0; i < words; ++i)
    {
        input_words[i] = i + 1;
        result_words[i] = 0xdead0000u + i;
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

static uint32_t verify_results(uint32_t words)
{
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < words; ++i)
    {
        if (result_words[i] == expected_words[i])
            continue;

        if (mismatches < 8)
        {
            printk("ERROR: i=%u got=%u expected=%u\n",
                   i, result_words[i], expected_words[i]);
        }
        ++mismatches;
    }
    return mismatches;
}

void notmain(void)
{
    struct vc4_runtime rt;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    uint32_t activeQpus = vc4_runtime_active_qpus(&rt);
    uint32_t laneWidth = vc4_runtime_lane_width();
    uint32_t words = activeQpus * laneWidth;

    if (laneWidth != TMU_READ_NOP_WRITE_WORDS_PER_QPU)
        panic("Unexpected lane width: %u", laneWidth);
    if (words > TMU_READ_NOP_WRITE_MAX_WORDS)
        panic("Unexpected word count: %u", words);

    fill_inputs(words);

    printk("Running VC4 tmu_read_nop_write reference bundle...\n");
    int start = timer_get_usec();
    if (tmu_read_nop_write_launch(&rt, input_words, result_words, words) < 0)
        panic("tmu_read_nop_write launch failed");
    int end = timer_get_usec();

    uint32_t mismatches = verify_results(words);
    uint32_t checksum = checksum_words(result_words, words);

    printk("tmu_read_nop_write qpus=%u lanes=%u words=%u checksum=%u\n",
           activeQpus, laneWidth, words, checksum);
    for (uint32_t i = 0; i < 4 && i < words; ++i)
    {
        printk("sample i=%u input=%u result=%u expected=%u\n",
               i, input_words[i], result_words[i], expected_words[i]);
    }

    printk("VC4_TEST_RESULT name=tmu_read_nop_write status=%s "
           "mismatches=%u active_qpus=%u words=%u checksum=%u elapsed_usec=%d\n",
           mismatches ? "FAIL" : "PASS",
           mismatches,
           activeQpus,
           words,
           checksum,
           end - start);

    if (mismatches)
        panic("tmu_read_nop_write verification failed: mismatches=%u", mismatches);

    vc4_runtime_shutdown(&rt);
}
