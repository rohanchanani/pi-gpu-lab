#include "rpi.h"
#include "qpu_num_register_launch.h"

static uint32_t qpu_num_results[QPU_NUM_REGISTER_LAUNCHES]
    __attribute__((aligned(16)));

static void clear_results(void)
{
    for (uint32_t i = 0; i < QPU_NUM_REGISTER_LAUNCHES; i++)
        qpu_num_results[i] = QPU_NUM_REGISTER_SENTINEL;
}

void notmain(void)
{
    struct vc4_runtime rt;

    clear_results();

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    printk("Running VC4 qpu_num_register reference bundle...\n");
    printk("Requesting %d QPU user programs; runtime active_qpus=%d lanes=%d\n",
           (int)QPU_NUM_REGISTER_LAUNCHES,
           (int)vc4_runtime_active_qpus(&rt),
           (int)vc4_runtime_lane_width());

    int start = timer_get_usec();
    if (qpu_num_register_launch(&rt, qpu_num_results) < 0)
        panic("qpu_num_register launch failed");
    int end = timer_get_usec();
    int elapsed = end - start;

    uint32_t pending_results = 0;
    uint32_t invalid_results = 0;
    uint32_t qpu_mask = 0;

    for (uint32_t i = 0; i < QPU_NUM_REGISTER_LAUNCHES; i++)
    {
        uint32_t value = qpu_num_results[i];
        printk("Launch %d QPU_NUM: %d\n", (int)(i + 1), (int)value);

        if (value == QPU_NUM_REGISTER_SENTINEL)
        {
            pending_results++;
        }
        else if (value > 15)
        {
            invalid_results++;
        }
        else
        {
            qpu_mask |= 1u << value;
        }
    }

    if (pending_results || invalid_results)
        panic("qpu_num_register verification failed: pending=%d invalid=%d",
              (int)pending_results,
              (int)invalid_results);

    printk("VC4_TEST_RESULT name=qpu_num_register status=PASS launches=%d completed_requests=%d invalid_results=%d pending_results=%d qpu_mask=%d active_qpus=%d elapsed_usec=%d\n",
           (int)QPU_NUM_REGISTER_LAUNCHES,
           (int)QPU_NUM_REGISTER_LAUNCHES,
           (int)invalid_results,
           (int)pending_results,
           (int)qpu_mask,
           (int)vc4_runtime_active_qpus(&rt),
           elapsed);

    vc4_runtime_shutdown(&rt);
}
