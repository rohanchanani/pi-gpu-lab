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

    printk("Running VC4 qpu_num_register targeted reservation test...\n");
    printk("Each result slot is indexed by target physical QPU, not launch order.\n");
    printk("runtime active_qpus=%d lanes=%d\n",
           (int)vc4_runtime_active_qpus(&rt),
           (int)vc4_runtime_lane_width());

    int start = timer_get_usec();
    if (qpu_num_register_launch(&rt, qpu_num_results) < 0)
        panic("qpu_num_register launch failed");
    int end = timer_get_usec();
    int elapsed = end - start;

    const struct qpu_num_register_debug *dbg = qpu_num_register_last_debug();

    uint32_t vpm_kbytes = (dbg->ident1 >> 28) & 0xfu;
    if (vpm_kbytes == 0)
        vpm_kbytes = 16;

    printk("V3D_IDENT0=%x IDENT1=%x IDENT2=%x\n",
           (unsigned)dbg->ident0,
           (unsigned)dbg->ident1,
           (unsigned)dbg->ident2);
    printk("V3D topology: VPMSZ_KiB=%d QUPS=%d NSLC=%d detected_active_qpus=%d expected_mask=%x\n",
           (int)vpm_kbytes,
           (int)dbg->qpus_per_slice,
           (int)dbg->num_slices,
           (int)dbg->active_qpus,
           (unsigned)dbg->expected_mask);
    printk("V3D before: SQRSV0=%x SQRSV1=%x VPMBASE=%x DBQITE=%x ERRSTAT=%x\n",
           (unsigned)dbg->sqrsv0_before,
           (unsigned)dbg->sqrsv1_before,
           (unsigned)dbg->vpm_base_before,
           (unsigned)dbg->dbqite_before,
           (unsigned)dbg->errstat_before);
    printk("V3D after:  SQRSV0=%x SQRSV1=%x VPMBASE=%x DBQITE=%x SRQCS=%x ERRSTAT=%x\n",
           (unsigned)dbg->sqrsv0_after,
           (unsigned)dbg->sqrsv1_after,
           (unsigned)dbg->vpm_base_after,
           (unsigned)dbg->dbqite_after,
           (unsigned)dbg->srqcs_after,
           (unsigned)dbg->errstat_after);
    printk("Observed masks: qpu_num_seen_mask=%x host_irq_mask=%x\n",
           (unsigned)dbg->seen_mask,
           (unsigned)dbg->irq_mask);

    uint32_t pending_results = 0;
    uint32_t invalid_results = 0;
    uint32_t mismatch_results = 0;
    uint32_t absent_results = 0;
    uint32_t qpu_mask = 0;

    for (uint32_t q = 0; q < QPU_NUM_REGISTER_LAUNCHES; q++)
    {
        uint32_t value = qpu_num_results[q];

        if (value == QPU_NUM_REGISTER_ABSENT)
        {
            printk("Target QPU %d: absent/skipped\n", (int)q);
            absent_results++;
            continue;
        }

        printk("Target QPU %d observed QPU_NUM: %d\n", (int)q, (int)value);

        if (value == QPU_NUM_REGISTER_SENTINEL)
        {
            pending_results++;
        }
        else if (value >= QPU_NUM_REGISTER_MAX_QPUS)
        {
            invalid_results++;
        }
        else
        {
            qpu_mask |= 1u << value;
            if (q < dbg->active_qpus && value != q)
                mismatch_results++;
        }
    }

    if (pending_results || invalid_results || mismatch_results ||
        ((qpu_mask & dbg->expected_mask) != dbg->expected_mask))
    {
        panic("qpu_num_register failed: pending=%d invalid=%d mismatch=%d qpu_mask=%x expected=%x",
              (int)pending_results,
              (int)invalid_results,
              (int)mismatch_results,
              (unsigned)qpu_mask,
              (unsigned)dbg->expected_mask);
    }

    printk("VC4_TEST_RESULT name=qpu_num_register status=PASS targets=%d absent=%d pending_results=%d invalid_results=%d mismatch_results=%d qpu_mask=%x irq_mask=%x expected_mask=%x elapsed_usec=%d\n",
           (int)dbg->active_qpus,
           (int)absent_results,
           (int)pending_results,
           (int)invalid_results,
           (int)mismatch_results,
           (unsigned)qpu_mask,
           (unsigned)dbg->irq_mask,
           (unsigned)dbg->expected_mask,
           elapsed);

    vc4_runtime_shutdown(&rt);
}
