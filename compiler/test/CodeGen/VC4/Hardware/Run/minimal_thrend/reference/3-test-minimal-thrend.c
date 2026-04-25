#include "rpi.h"
#include "minimal_thrend_launch.h"

void notmain(void)
{
    struct vc4_runtime rt;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    uint32_t activeQpus = vc4_runtime_active_qpus(&rt);
    printk("Running VC4 minimal_thrend reference bundle...\n");

    int start_time = timer_get_usec();
    if (minimal_thrend_launch(&rt) < 0)
        panic("minimal_thrend launch failed");
    int end_time = timer_get_usec();

    printk("minimal_thrend qpus=%d lanes=%d\n",
           activeQpus,
           vc4_runtime_lane_width());
    printk("VC4_TEST_RESULT name=minimal_thrend status=PASS completed_qpus=%d active_qpus=%d elapsed_usec=%d\n",
           activeQpus,
           activeQpus,
           end_time - start_time);

    vc4_runtime_shutdown(&rt);
}
