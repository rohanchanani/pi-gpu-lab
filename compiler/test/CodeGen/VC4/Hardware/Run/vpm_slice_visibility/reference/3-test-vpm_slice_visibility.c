#include "rpi.h"
#include "vpm_slice_visibility_launch.h"

#define ERRSTAT_RELEVANT_MASK 0x0000efffu /* exclude VCD idle bit 12 */

static struct vpm_slice_visibility_results results
    __attribute__((aligned(16)));

static uint32_t tag_sanity(uint32_t lane)
{
    return 0xd5000000u | (lane & 0x0fu);
}

static uint32_t tag_visibility(uint32_t writer_qpu,
                               uint32_t trial,
                               uint32_t lane)
{
    return 0xa5000000u |
           ((trial & 0x0fu) << 20) |
           ((writer_qpu & 0x0fu) << 8) |
           (lane & 0x0fu);
}

static uint32_t tag_collision(uint32_t writer_qpu,
                              uint32_t order,
                              uint32_t lane)
{
    return 0xc5000000u |
           ((order & 0x0fu) << 20) |
           ((writer_qpu & 0x0fu) << 8) |
           (lane & 0x0fu);
}

static uint32_t vector_matches_visibility(
    const uint32_t vec[VPM_SLICE_VISIBILITY_LANES],
    uint32_t writer_qpu,
    uint32_t trial)
{
    for (uint32_t lane = 0; lane < VPM_SLICE_VISIBILITY_LANES; lane++)
    {
        if (vec[lane] != tag_visibility(writer_qpu, trial, lane))
            return 0;
    }
    return 1;
}

static uint32_t vector_matches_collision(
    const uint32_t vec[VPM_SLICE_VISIBILITY_LANES],
    uint32_t writer_qpu,
    uint32_t order)
{
    for (uint32_t lane = 0; lane < VPM_SLICE_VISIBILITY_LANES; lane++)
    {
        if (vec[lane] != tag_collision(writer_qpu, order, lane))
            return 0;
    }
    return 1;
}

static uint32_t vector_matches_sanity(
    const uint32_t vec[VPM_SLICE_VISIBILITY_LANES])
{
    for (uint32_t lane = 0; lane < VPM_SLICE_VISIBILITY_LANES; lane++)
    {
        if (vec[lane] != tag_sanity(lane))
            return 0;
    }
    return 1;
}

static uint32_t qpu_report_mismatches(
    const uint32_t vec[VPM_SLICE_VISIBILITY_LANES],
    uint32_t expected_qpu)
{
    uint32_t mismatches = 0;
    for (uint32_t lane = 0; lane < VPM_SLICE_VISIBILITY_LANES; lane++)
    {
        if (vec[lane] != expected_qpu)
            mismatches++;
    }
    return mismatches;
}

static uint32_t errstat_relevant_changed(uint32_t before, uint32_t after)
{
    return (((before ^ after) & ERRSTAT_RELEVANT_MASK) != 0);
}

static void print_vector(const char *prefix,
                         const uint32_t vec[VPM_SLICE_VISIBILITY_LANES])
{
    printk("%s", prefix);
    for (uint32_t lane = 0; lane < VPM_SLICE_VISIBILITY_LANES; lane++)
        printk(" %x", vec[lane]);
    printk("\n");
}

static void print_topology_json(void)
{
    printk("vc4_vpm_topology.json:\n");
    printk("{\n");
    printk("  ident1: %x,\n", results.ident1);
    printk("  vpmsz_field: %d,\n", (int)results.vpmsz_field);
    printk("  vpm_kib: %d,\n", (int)results.vpm_kib);
    printk("  tmus_per_slice: %d,\n", (int)results.tmus_per_slice);
    printk("  qpus_per_slice: %d,\n", (int)results.qpus_per_slice);
    printk("  num_slices: %d,\n", (int)results.num_slices);
    printk("  revision: %d,\n", (int)results.revision);
    printk("  num_qpus: %d,\n", (int)results.num_qpus);
    printk("  vpmbase_written: %d,\n", (int)results.vpmbase_written);
    printk("  vpmbase_readback: %x\n", results.vpmbase_readback);
    printk("}\n");
}

static uint32_t validate_sanity(uint32_t *reported_qpu_mismatches,
                                uint32_t *errstat_changed)
{
    uint32_t sanity_mismatches = 0;

    if (!vector_matches_sanity(results.sanity.observed))
        sanity_mismatches = VPM_SLICE_VISIBILITY_LANES;

    *reported_qpu_mismatches +=
        qpu_report_mismatches(results.sanity.qpu_report, 0);

    *errstat_changed += errstat_relevant_changed(results.sanity.errstat_before,
                                                 results.sanity.errstat_after);

    printk("Single-QPU sanity: qpu_report_lane0=%d observed_match=%d errstat_before=%x errstat_after=%x timeout=%d\n",
           (int)results.sanity.qpu_report[0],
           (int)(sanity_mismatches == 0),
           results.sanity.errstat_before,
           results.sanity.errstat_after,
           (int)results.sanity.timeout);

    if (sanity_mismatches)
    {
        print_vector("SANITY observed:", results.sanity.observed);
        printk("SANITY expected lane0=%x lane15=%x\n",
               tag_sanity(0),
               tag_sanity(15));
    }

    return sanity_mismatches;
}

static void analyze_visibility(uint32_t *same_slice_visibility_pass,
                               uint32_t *cross_slice_visibility_matches,
                               uint32_t *cross_slice_visibility_misses,
                               uint32_t *reported_qpu_mismatches,
                               uint32_t *errstat_changed)
{
    *same_slice_visibility_pass = 0;
    *cross_slice_visibility_matches = 0;
    *cross_slice_visibility_misses = 0;

    printk("vc4_vpm_visibility_matrix.csv:\n");
    printk("trial,writer,reader,writer_slice,reader_slice,match,observed0,expected0,errstat_before,errstat_after,timeout\n");

    for (uint32_t i = 0; i < results.visibility_pair_count; i++)
    {
        const struct vpm_slice_visibility_pair_result *pair =
            &results.visibility[i];
        uint32_t match = vector_matches_visibility(pair->observed,
                                                   pair->writer_qpu,
                                                   pair->trial);
        uint32_t same_slice = (pair->writer_slice == pair->reader_slice);
        uint32_t expected0 = tag_visibility(pair->writer_qpu,
                                            pair->trial,
                                            0);

        if (same_slice && match)
            *same_slice_visibility_pass = 1;
        if (!same_slice && match)
            (*cross_slice_visibility_matches)++;
        if (!same_slice && !match)
            (*cross_slice_visibility_misses)++;

        *reported_qpu_mismatches +=
            qpu_report_mismatches(pair->writer_report, pair->writer_qpu);
        *reported_qpu_mismatches +=
            qpu_report_mismatches(pair->reader_report, pair->reader_qpu);
        *errstat_changed += errstat_relevant_changed(pair->errstat_before,
                                                     pair->errstat_after);

        printk("%d,%d,%d,%d,%d,%d,%x,%x,%x,%x,%d\n",
               (int)pair->trial,
               (int)pair->writer_qpu,
               (int)pair->reader_qpu,
               (int)pair->writer_slice,
               (int)pair->reader_slice,
               (int)match,
               pair->observed[0],
               expected0,
               pair->errstat_before,
               pair->errstat_after,
               (int)pair->timeout);

        printk("VISIBILITY_PAIR writer=%d reader=%d writer_slice=%d reader_slice=%d match=%d\n",
               (int)pair->writer_qpu,
               (int)pair->reader_qpu,
               (int)pair->writer_slice,
               (int)pair->reader_slice,
               (int)match);

        if (!match)
        {
            print_vector("  observed:", pair->observed);
            printk("  expected lane0=%x lane15=%x\n",
                   expected0,
                   tag_visibility(pair->writer_qpu, pair->trial, 15));
        }
    }
}

static void analyze_collision(uint32_t *same_slice_collision_pass,
                              uint32_t *cross_collision_global_like,
                              uint32_t *cross_collision_per_slice_like,
                              uint32_t *cross_collision_other,
                              uint32_t *reported_qpu_mismatches,
                              uint32_t *errstat_changed)
{
    uint32_t same_slice_runs = 0;
    uint32_t same_slice_runs_pass = 0;

    *cross_collision_global_like = 0;
    *cross_collision_per_slice_like = 0;
    *cross_collision_other = 0;

    printk("vc4_vpm_collision_matrix.csv:\n");
    printk("trial,a,b,slice_a,slice_b,order,a_reads_a,a_reads_b,b_reads_a,b_reads_b,class,errstat_before,errstat_after,timeout\n");

    for (uint32_t i = 0; i < results.collision_run_count; i++)
    {
        const struct vpm_slice_visibility_collision_result *collision =
            &results.collision[i];
        uint32_t a_reads_a = vector_matches_collision(collision->observed_by_a,
                                                      collision->qpu_a,
                                                      collision->order);
        uint32_t a_reads_b = vector_matches_collision(collision->observed_by_a,
                                                      collision->qpu_b,
                                                      collision->order);
        uint32_t b_reads_a = vector_matches_collision(collision->observed_by_b,
                                                      collision->qpu_a,
                                                      collision->order);
        uint32_t b_reads_b = vector_matches_collision(collision->observed_by_b,
                                                      collision->qpu_b,
                                                      collision->order);
        uint32_t same_slice = collision->slice_a == collision->slice_b;
        uint32_t last_writer = collision->order == 0 ? collision->qpu_b
                                                     : collision->qpu_a;
        uint32_t both_read_last = 0;
        uint32_t each_reads_own = a_reads_a && b_reads_b;
        uint32_t class_id = 0;

        if (last_writer == collision->qpu_a)
            both_read_last = a_reads_a && b_reads_a;
        else
            both_read_last = a_reads_b && b_reads_b;

        if (same_slice)
        {
            same_slice_runs++;
            if (both_read_last)
                same_slice_runs_pass++;
        }
        else if (both_read_last)
        {
            (*cross_collision_global_like)++;
            class_id = 1;
        }
        else if (each_reads_own)
        {
            (*cross_collision_per_slice_like)++;
            class_id = 2;
        }
        else
        {
            (*cross_collision_other)++;
            class_id = 3;
        }

        *reported_qpu_mismatches +=
            qpu_report_mismatches(collision->qpu_a_report, collision->qpu_a);
        *reported_qpu_mismatches +=
            qpu_report_mismatches(collision->qpu_b_report, collision->qpu_b);
        *errstat_changed += errstat_relevant_changed(collision->errstat_before,
                                                     collision->errstat_after);

        printk("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%x,%x,%d\n",
               (int)collision->trial,
               (int)collision->qpu_a,
               (int)collision->qpu_b,
               (int)collision->slice_a,
               (int)collision->slice_b,
               (int)collision->order,
               (int)a_reads_a,
               (int)a_reads_b,
               (int)b_reads_a,
               (int)b_reads_b,
               (int)class_id,
               collision->errstat_before,
               collision->errstat_after,
               (int)collision->timeout);

        printk("COLLISION_PAIR a=%d b=%d order=%d same_slice=%d both_read_last=%d each_reads_own=%d class=%d\n",
               (int)collision->qpu_a,
               (int)collision->qpu_b,
               (int)collision->order,
               (int)same_slice,
               (int)both_read_last,
               (int)each_reads_own,
               (int)class_id);

        if (!same_slice && class_id == 3)
        {
            print_vector("  observed_by_a:", collision->observed_by_a);
            print_vector("  observed_by_b:", collision->observed_by_b);
        }
    }

    *same_slice_collision_pass =
        (same_slice_runs != 0 && same_slice_runs_pass == same_slice_runs);
}

void notmain(void)
{
    struct vc4_runtime rt;

    if (vc4_runtime_init(&rt) < 0)
        panic("Failed to initialize VC4 runtime");

    printk("Running VC4 vpm_slice_visibility reference bundle...\n");
    int start = timer_get_usec();
    int launch_rc = vpm_slice_visibility_launch(&rt, &results);
    int end = timer_get_usec();
    int elapsed = end - start;

    print_topology_json();

    uint32_t invalid_topology = 0;
    if (results.qpus_per_slice < 2 ||
        results.num_slices < 2 ||
        results.num_qpus == 0 ||
        results.num_qpus > VPM_SLICE_VISIBILITY_MAX_QPUS)
        invalid_topology = 1;

    uint32_t reported_qpu_mismatches = 0;
    uint32_t errstat_relevant_changed_count = 0;
    uint32_t sanity_mismatches = validate_sanity(&reported_qpu_mismatches,
                                                 &errstat_relevant_changed_count);

    uint32_t same_slice_visibility_pass = 0;
    uint32_t cross_slice_visibility_matches = 0;
    uint32_t cross_slice_visibility_misses = 0;
    analyze_visibility(&same_slice_visibility_pass,
                       &cross_slice_visibility_matches,
                       &cross_slice_visibility_misses,
                       &reported_qpu_mismatches,
                       &errstat_relevant_changed_count);

    uint32_t same_slice_collision_pass = 0;
    uint32_t cross_collision_global_like = 0;
    uint32_t cross_collision_per_slice_like = 0;
    uint32_t cross_collision_other = 0;
    analyze_collision(&same_slice_collision_pass,
                      &cross_collision_global_like,
                      &cross_collision_per_slice_like,
                      &cross_collision_other,
                      &reported_qpu_mismatches,
                      &errstat_relevant_changed_count);

    uint32_t conclusion_id = 0;
    if (cross_slice_visibility_matches != 0 &&
        cross_slice_visibility_misses == 0 &&
        cross_collision_global_like != 0 &&
        cross_collision_per_slice_like == 0 &&
        cross_collision_other == 0)
    {
        conclusion_id = 1; /* global-like */
        printk("VPM storage interpretation: global_like_user_visible_4KiB_window\n");
    }
    else if (cross_slice_visibility_misses != 0 &&
             cross_slice_visibility_matches == 0 &&
             cross_collision_per_slice_like != 0 &&
             cross_collision_global_like == 0 &&
             cross_collision_other == 0)
    {
        conclusion_id = 2; /* per-slice-like */
        printk("VPM storage interpretation: per_slice_like_4KiB_windows\n");
    }
    else
    {
        conclusion_id = 3; /* mixed or inconclusive */
        printk("VPM storage interpretation: mixed_or_inconclusive\n");
    }

    uint32_t pass = 1;
    if (launch_rc < 0)
        pass = 0;
    if (invalid_topology)
        pass = 0;
    if (sanity_mismatches != 0)
        pass = 0;
    if (!same_slice_visibility_pass)
        pass = 0;
    if (!same_slice_collision_pass)
        pass = 0;
    if (reported_qpu_mismatches != 0)
        pass = 0;
    if (results.timeouts != 0)
        pass = 0;
    if (errstat_relevant_changed_count != 0)
        pass = 0;

    printk("vc4_vpm_full_log.txt: elapsed_usec=%d launch_rc=%d srqcs_after_last_run=%x\n",
           elapsed,
           launch_rc,
           results.srqcs_after_last_run);

    if (pass)
    {
        printk("VC4_TEST_RESULT name=vpm_slice_visibility status=PASS sanity_mismatches=%d same_slice_visibility_pass=%d same_slice_collision_pass=%d reported_qpu_mismatches=%d timeouts=%d invalid_topology=%d errstat_relevant_changed=%d visibility_pairs=%d collision_runs=%d cross_slice_visibility_matches=%d cross_slice_visibility_misses=%d cross_collision_global_like=%d cross_collision_per_slice_like=%d cross_collision_other=%d conclusion_id=%d elapsed_usec=%d\n",
               (int)sanity_mismatches,
               (int)same_slice_visibility_pass,
               (int)same_slice_collision_pass,
               (int)reported_qpu_mismatches,
               (int)results.timeouts,
               (int)invalid_topology,
               (int)errstat_relevant_changed_count,
               (int)results.visibility_pair_count,
               (int)results.collision_run_count,
               (int)cross_slice_visibility_matches,
               (int)cross_slice_visibility_misses,
               (int)cross_collision_global_like,
               (int)cross_collision_per_slice_like,
               (int)cross_collision_other,
               (int)conclusion_id,
               elapsed);
    }
    else
    {
        printk("VC4_TEST_RESULT name=vpm_slice_visibility status=FAIL sanity_mismatches=%d same_slice_visibility_pass=%d same_slice_collision_pass=%d reported_qpu_mismatches=%d timeouts=%d invalid_topology=%d errstat_relevant_changed=%d visibility_pairs=%d collision_runs=%d cross_slice_visibility_matches=%d cross_slice_visibility_misses=%d cross_collision_global_like=%d cross_collision_per_slice_like=%d cross_collision_other=%d conclusion_id=%d elapsed_usec=%d\n",
               (int)sanity_mismatches,
               (int)same_slice_visibility_pass,
               (int)same_slice_collision_pass,
               (int)reported_qpu_mismatches,
               (int)results.timeouts,
               (int)invalid_topology,
               (int)errstat_relevant_changed_count,
               (int)results.visibility_pair_count,
               (int)results.collision_run_count,
               (int)cross_slice_visibility_matches,
               (int)cross_slice_visibility_misses,
               (int)cross_collision_global_like,
               (int)cross_collision_per_slice_like,
               (int)cross_collision_other,
               (int)conclusion_id,
               elapsed);
        panic("vpm_slice_visibility validation failed");
    }

    vc4_runtime_shutdown(&rt);
}
