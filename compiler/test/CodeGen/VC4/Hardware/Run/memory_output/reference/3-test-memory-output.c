#include "rpi.h"
#include <stdint.h>
#include <string.h>
#include "memory_output_launch.h"

static uint32_t gpu_result_words[MEMORY_OUTPUT_MAX_WORDS];

static uint32_t checksum_words(const uint32_t *values, uint32_t words)
{
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < words; i++)
        checksum += values[i];
    return checksum;
}

static int verify_output(uint32_t active_qpus, uint32_t *first_bad_index,
                         uint32_t *first_bad_value,
                         uint32_t *first_bad_expected)
{
    int mismatches = 0;
    uint32_t words_per_qpu = MEMORY_OUTPUT_WORDS_PER_QPU;

    *first_bad_index = 0;
    *first_bad_value = 0;
    *first_bad_expected = 0;

    for (uint32_t qpu = 0; qpu < active_qpus; qpu++)
    {
        for (uint32_t lane = 0; lane < words_per_qpu; lane++)
        {
            uint32_t index = qpu * words_per_qpu + lane;
            uint32_t expected = qpu;
            uint32_t actual = gpu_result_words[index];
            if (actual != expected)
            {
                if (mismatches == 0)
                {
                    *first_bad_index = index;
                    *first_bad_value = actual;
                    *first_bad_expected = expected;
                }
                if (mismatches < 8)
                {
                    printk("ERROR: index=%d actual=%d expected=%d\n",
                           index, actual, expected);
                }
                mismatches++;
            }
        }
    }

    return mismatches;
}

void notmain(void)
{
    struct vc4_runtime rt;
    uint32_t active_qpus;
    uint32_t words;
    uint32_t checksum;
    uint32_t first_bad_index = 0;
    uint32_t first_bad_value = 0;
    uint32_t first_bad_expected = 0;
    int mismatches;
    int start_time;
    int end_time;
    int elapsed_usec;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    active_qpus = vc4_runtime_active_qpus(&rt);
    words = active_qpus * MEMORY_OUTPUT_WORDS_PER_QPU;

    memset(gpu_result_words, 0, sizeof gpu_result_words);

    printk("Running VC4 memory_output reference bundle...\n");
    start_time = timer_get_usec();
    if (memory_output_launch(&rt, gpu_result_words, MEMORY_OUTPUT_MAX_WORDS) < 0)
        panic("memory_output launch failed");
    end_time = timer_get_usec();
    elapsed_usec = end_time - start_time;

    mismatches = verify_output(active_qpus, &first_bad_index,
                               &first_bad_value, &first_bad_expected);
    checksum = checksum_words(gpu_result_words, words);

    printk("memory_output qpus=%d lanes=%d words=%d checksum=%d\n",
           active_qpus, vc4_runtime_lane_width(), words, checksum);
    printk("memory_output sample: out[0]=%d out[16]=%d out[32]=%d\n",
           gpu_result_words[0], gpu_result_words[16], gpu_result_words[32]);

    printk("VC4_TEST_RESULT name=memory_output status=%s mismatches=%d "
           "active_qpus=%d words=%d checksum=%d elapsed_usec=%d\n",
           mismatches ? "FAIL" : "PASS",
           mismatches,
           active_qpus,
           words,
           checksum,
           elapsed_usec);

    if (mismatches)
    {
        panic("memory_output verification failed: first_bad_index=%d "
              "actual=%d expected=%d",
              first_bad_index, first_bad_value, first_bad_expected);
    }

    printk("SUCCESS: memory_output verified.\n");
    vc4_runtime_shutdown(&rt);
}
