#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ERRSTAT_RELEVANT_MASK 0x0000efffu
#define VPM_SETUP_CLOBBER_LANES 16u
#define VPM_SETUP_CLOBBER_MAX_QPUS 16u
#define VPM_SETUP_CLOBBER_QPUS_PER_SLICE 4u
#define VPM_SETUP_CLOBBER_NUM_SLICES 3u
#define VPM_SETUP_CLOBBER_NUM_QPUS 12u
#define VPM_SETUP_CLOBBER_NUM_SEMAPHORES 16u
#define VPM_SETUP_CLOBBER_VPM_URSV_4K 16u
#define VPM_SETUP_CLOBBER_CLOBBER_ROWA 17u
#define VPM_SETUP_CLOBBER_CLOBBER_ROWB 18u
#define VPM_SETUP_CLOBBER_SENTINEL 0xdeadbeefu
#define VPM_SETUP_CLOBBER_TRIALS_PER_PAIR 3u
#define VPM_SETUP_CLOBBER_MAX_RECORDS 32u

#define VPM_SETUP_CLOBBER_CLASS_NO_CLOBBER 0u
#define VPM_SETUP_CLOBBER_CLASS_SETUP_CLOBBER 1u
#define VPM_SETUP_CLOBBER_CLASS_BOTH 2u
#define VPM_SETUP_CLOBBER_CLASS_OTHER 3u

struct qpu_pair {
    uint32_t a;
    uint32_t b;
};

struct vpm_setup_clobber_record {
    uint32_t trial;
    uint32_t qpu_a;
    uint32_t qpu_b;
    uint32_t slice_a;
    uint32_t slice_b;

    uint32_t row_a[VPM_SETUP_CLOBBER_LANES];
    uint32_t row_b[VPM_SETUP_CLOBBER_LANES];
    uint32_t qpu_a_report[VPM_SETUP_CLOBBER_LANES];
    uint32_t qpu_b_report[VPM_SETUP_CLOBBER_LANES];

    uint32_t row_a_match;
    uint32_t row_b_match;
    uint32_t classification;
    uint32_t qpu_mismatches;

    uint32_t errstat_before;
    uint32_t errstat_after;
    uint32_t timeout;
};

static struct vpm_setup_clobber_record records[VPM_SETUP_CLOBBER_MAX_RECORDS];
static uint32_t row_a_values[VPM_SETUP_CLOBBER_LANES];
static uint32_t row_b_values[VPM_SETUP_CLOBBER_LANES];
static uint32_t qpu_a_report_values[VPM_SETUP_CLOBBER_LANES];
static uint32_t qpu_b_report_values[VPM_SETUP_CLOBBER_LANES];

static uint32_t errstat_relevant_changed(uint32_t before, uint32_t after)
{
    return (((before ^ after) & ERRSTAT_RELEVANT_MASK) != 0);
}

static uint32_t slice_of_qpu(uint32_t qpu)
{
    return qpu / VPM_SETUP_CLOBBER_QPUS_PER_SLICE;
}

static uint32_t expected_tag(uint32_t qpu_a, uint32_t lane)
{
    return 0xe5000000u | ((qpu_a & 0x0fu) << 8) | (lane & 0x0fu);
}

static void fill_vector(uint32_t vec[VPM_SETUP_CLOBBER_LANES], uint32_t value)
{
    for (uint32_t i = 0; i < VPM_SETUP_CLOBBER_LANES; i++)
        vec[i] = value;
}

static void copy_vector(uint32_t dst[VPM_SETUP_CLOBBER_LANES],
                        const uint32_t src[VPM_SETUP_CLOBBER_LANES])
{
    for (uint32_t i = 0; i < VPM_SETUP_CLOBBER_LANES; i++)
        dst[i] = src[i];
}

static uint32_t vector_matches_tag(
    const uint32_t vec[VPM_SETUP_CLOBBER_LANES],
    uint32_t qpu_a)
{
    for (uint32_t lane = 0; lane < VPM_SETUP_CLOBBER_LANES; lane++) {
        if (vec[lane] != expected_tag(qpu_a, lane))
            return 0;
    }
    return 1;
}

static uint32_t qpu_report_mismatches(
    const uint32_t vec[VPM_SETUP_CLOBBER_LANES],
    uint32_t expected_qpu)
{
    uint32_t mismatches = 0;
    for (uint32_t lane = 0; lane < VPM_SETUP_CLOBBER_LANES; lane++) {
        if (vec[lane] != expected_qpu)
            mismatches++;
    }
    return mismatches;
}

static uint32_t classify_record(struct vpm_setup_clobber_record *record)
{
    record->row_a_match = vector_matches_tag(record->row_a, record->qpu_a);
    record->row_b_match = vector_matches_tag(record->row_b, record->qpu_a);

    if (record->row_a_match && !record->row_b_match)
        return VPM_SETUP_CLOBBER_CLASS_NO_CLOBBER;
    if (!record->row_a_match && record->row_b_match)
        return VPM_SETUP_CLOBBER_CLASS_SETUP_CLOBBER;
    if (record->row_a_match && record->row_b_match)
        return VPM_SETUP_CLOBBER_CLASS_BOTH;
    return VPM_SETUP_CLOBBER_CLASS_OTHER;
}

static const char *classification_name(uint32_t classification)
{
    switch (classification) {
    case VPM_SETUP_CLOBBER_CLASS_NO_CLOBBER:
        return "no_clobber";
    case VPM_SETUP_CLOBBER_CLASS_SETUP_CLOBBER:
        return "setup_clobber";
    case VPM_SETUP_CLOBBER_CLASS_BOTH:
        return "both";
    case VPM_SETUP_CLOBBER_CLASS_OTHER:
        return "other";
    default:
        return "unknown";
    }
}

static uint32_t append_pair(
    struct qpu_pair pairs[VPM_SETUP_CLOBBER_MAX_RECORDS],
    uint32_t count,
    uint32_t a,
    uint32_t b)
{
    if (count >= VPM_SETUP_CLOBBER_MAX_RECORDS)
        return count;
    pairs[count].a = a;
    pairs[count].b = b;
    return count + 1u;
}

static uint32_t build_representative_pairs(
    struct qpu_pair pairs[VPM_SETUP_CLOBBER_MAX_RECORDS])
{
    uint32_t count = 0;
    uint32_t qps = VPM_SETUP_CLOBBER_QPUS_PER_SLICE;
    uint32_t nslices = VPM_SETUP_CLOBBER_NUM_SLICES;
    uint32_t nq = VPM_SETUP_CLOBBER_NUM_QPUS;

    if (nq >= 2u) {
        count = append_pair(pairs, count, 0u, 1u);
        count = append_pair(pairs, count, 1u, 0u);
    }

    if (qps != 0u && nslices >= 2u && qps < nq) {
        count = append_pair(pairs, count, 0u, qps);
        count = append_pair(pairs, count, qps, 0u);
    }

    if (qps != 0u && nslices >= 3u && (2u * qps) < nq) {
        count = append_pair(pairs, count, 0u, 2u * qps);
        count = append_pair(pairs, count, 2u * qps, 0u);
    }

    return count;
}

static int run_pair(struct vc4_program *program,
                    vc4_deviceptr_t row_a_dev,
                    vc4_deviceptr_t row_b_dev,
                    vc4_deviceptr_t qpu_a_report_dev,
                    vc4_deviceptr_t qpu_b_report_dev,
                    uint32_t qpu_a,
                    uint32_t qpu_b,
                    uint32_t trial,
                    struct vpm_setup_clobber_record *record)
{
    for (uint32_t i = 0; i < VPM_SETUP_CLOBBER_LANES; i++) {
        row_a_values[i] = VPM_SETUP_CLOBBER_SENTINEL;
        row_b_values[i] = VPM_SETUP_CLOBBER_SENTINEL;
        qpu_a_report_values[i] = VPM_SETUP_CLOBBER_SENTINEL;
        qpu_b_report_values[i] = VPM_SETUP_CLOBBER_SENTINEL;
    }

    record->trial = trial;
    record->qpu_a = qpu_a;
    record->qpu_b = qpu_b;
    record->slice_a = slice_of_qpu(qpu_a);
    record->slice_b = slice_of_qpu(qpu_b);
    record->errstat_before = 0;
    record->errstat_after = 0;
    record->timeout = 0;

    uint32_t bytes = VPM_SETUP_CLOBBER_LANES * sizeof(uint32_t);
    if (vc4_m2_copy_htod(program, row_a_dev, row_a_values, bytes) < 0 ||
        vc4_m2_copy_htod(program, row_b_dev, row_b_values, bytes) < 0 ||
        vc4_m2_copy_htod(program, qpu_a_report_dev, qpu_a_report_values, bytes) < 0 ||
        vc4_m2_copy_htod(program, qpu_b_report_dev, qpu_b_report_values, bytes) < 0) {
        record->timeout = 1;
        return -1;
    }

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(VPM_SETUP_CLOBBER_NUM_QPUS, 1, 1);
    int rc = vpm_setup_clobber_run_pair_launch(
        program,
        grid,
        block,
        qpu_a,
        qpu_b,
        trial,
        VPM_SETUP_CLOBBER_CLOBBER_ROWA,
        VPM_SETUP_CLOBBER_CLOBBER_ROWB,
        row_a_dev,
        row_b_dev,
        qpu_a_report_dev,
        qpu_b_report_dev);

    if (rc < 0)
        record->timeout = 1;

    if (vc4_m2_copy_dtoh(program, row_a_values, row_a_dev, bytes) < 0 ||
        vc4_m2_copy_dtoh(program, row_b_values, row_b_dev, bytes) < 0 ||
        vc4_m2_copy_dtoh(program, qpu_a_report_values, qpu_a_report_dev, bytes) < 0 ||
        vc4_m2_copy_dtoh(program, qpu_b_report_values, qpu_b_report_dev, bytes) < 0) {
        record->timeout = 1;
        return -1;
    }

    copy_vector(record->row_a, row_a_values);
    copy_vector(record->row_b, row_b_values);
    copy_vector(record->qpu_a_report, qpu_a_report_values);
    copy_vector(record->qpu_b_report, qpu_b_report_values);

    record->qpu_mismatches =
        qpu_report_mismatches(record->qpu_a_report, qpu_a) +
        qpu_report_mismatches(record->qpu_b_report, qpu_b);
    record->classification = classify_record(record);

    return rc < 0 ? -1 : 0;
}

static void print_topology_json(void)
{
    printk("vc4_vpm_setup_clobber_topology.json:\n");
    printk("{\n");
    printk(" ident1: 0,\n");
    printk(" vpmsz_field: %d,\n", (int)VPM_SETUP_CLOBBER_VPM_URSV_4K);
    printk(" vpm_kib: %d,\n", (int)VPM_SETUP_CLOBBER_VPM_URSV_4K);
    printk(" qpus_per_slice: %d,\n", (int)VPM_SETUP_CLOBBER_QPUS_PER_SLICE);
    printk(" num_slices: %d,\n", (int)VPM_SETUP_CLOBBER_NUM_SLICES);
    printk(" num_semaphores: %d,\n", (int)VPM_SETUP_CLOBBER_NUM_SEMAPHORES);
    printk(" num_qpus: %d,\n", (int)VPM_SETUP_CLOBBER_NUM_QPUS);
    printk(" vpmbase_written: %d,\n", (int)VPM_SETUP_CLOBBER_VPM_URSV_4K);
    printk(" vpmbase_readback: %x\n", VPM_SETUP_CLOBBER_VPM_URSV_4K);
    printk("}\n");
}

static void print_pair_diagnostic(
    const struct vpm_setup_clobber_record *record)
{
    printk("SETUP_CLOBBER_PAIR trial=%d a=%d b=%d slice_a=%d slice_b=%d row_a_match=%d row_b_match=%d classification=%s qpu_mismatches=%d timeout=%d errstat_before=%x errstat_after=%x row_a0=%x row_b0=%x expected0=%x\n",
           (int)record->trial,
           (int)record->qpu_a,
           (int)record->qpu_b,
           (int)record->slice_a,
           (int)record->slice_b,
           (int)record->row_a_match,
           (int)record->row_b_match,
           classification_name(record->classification),
           (int)record->qpu_mismatches,
           (int)record->timeout,
           record->errstat_before,
           record->errstat_after,
           record->row_a[0],
           record->row_b[0],
           expected_tag(record->qpu_a, 0u));
}

void notmain(void)
{
    struct vc4_program *program = 0;
    struct qpu_pair pairs[VPM_SETUP_CLOBBER_MAX_RECORDS];

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t row_a_dev = 0;
    vc4_deviceptr_t row_b_dev = 0;
    vc4_deviceptr_t qpu_a_report_dev = 0;
    vc4_deviceptr_t qpu_b_report_dev = 0;
    uint32_t bytes = VPM_SETUP_CLOBBER_LANES * sizeof(uint32_t);

    if (vc4_m2_malloc(program, &row_a_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &row_b_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &qpu_a_report_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &qpu_b_report_dev, bytes) < 0)
        panic("vpm_setup_clobber device allocation failed");

    printk("Running VC4 vpm_setup_clobber generated candidate...\n");
    print_topology_json();

    uint32_t invalid_topology = 0;
    if (VPM_SETUP_CLOBBER_QPUS_PER_SLICE != 4u)
        invalid_topology = 1;
    if (VPM_SETUP_CLOBBER_NUM_SLICES < 3u)
        invalid_topology = 1;
    if (VPM_SETUP_CLOBBER_NUM_QPUS < 12u)
        invalid_topology = 1;
    if (VPM_SETUP_CLOBBER_NUM_SEMAPHORES < 16u)
        invalid_topology = 1;

    uint32_t pair_count = build_representative_pairs(pairs);
    uint32_t representative_pairs =
        pair_count * VPM_SETUP_CLOBBER_TRIALS_PER_PAIR;

    uint32_t pairs_run = 0;
    uint32_t valid_pairs = 0;
    uint32_t setup_clobber_count = 0;
    uint32_t no_clobber_count = 0;
    uint32_t both_count = 0;
    uint32_t other_count = 0;
    uint32_t qpu_mismatches = 0;
    uint32_t timeouts = 0;
    uint32_t errstat_relevant_changed_count = 0;

    printk("vc4_vpm_setup_clobber_matrix.csv:\n");
    printk("trial,a,b,slice_a,slice_b,row_a_match,row_b_match,classification,errstat_before,errstat_after,timeout\n");

    int start = timer_get_usec();
    uint32_t stop_after_timeout = 0;

    if (!invalid_topology) {
        for (uint32_t i = 0; i < pair_count && !stop_after_timeout; i++) {
            for (uint32_t t = 0; t < VPM_SETUP_CLOBBER_TRIALS_PER_PAIR; t++) {
                if (pairs_run >= VPM_SETUP_CLOBBER_MAX_RECORDS) {
                    stop_after_timeout = 1;
                    break;
                }

                struct vpm_setup_clobber_record *record = &records[pairs_run];
                fill_vector(record->row_a, VPM_SETUP_CLOBBER_SENTINEL);
                fill_vector(record->row_b, VPM_SETUP_CLOBBER_SENTINEL);
                fill_vector(record->qpu_a_report, VPM_SETUP_CLOBBER_SENTINEL);
                fill_vector(record->qpu_b_report, VPM_SETUP_CLOBBER_SENTINEL);

                int rc = run_pair(program,
                                  row_a_dev,
                                  row_b_dev,
                                  qpu_a_report_dev,
                                  qpu_b_report_dev,
                                  pairs[i].a,
                                  pairs[i].b,
                                  t,
                                  record);
                (void)rc;
                pairs_run++;

                uint32_t err_changed =
                    errstat_relevant_changed(record->errstat_before,
                                             record->errstat_after);

                qpu_mismatches += record->qpu_mismatches;
                timeouts += record->timeout;
                errstat_relevant_changed_count += err_changed;

                if (!record->timeout &&
                    record->qpu_mismatches == 0 &&
                    err_changed == 0)
                    valid_pairs++;

                switch (record->classification) {
                case VPM_SETUP_CLOBBER_CLASS_NO_CLOBBER:
                    no_clobber_count++;
                    break;
                case VPM_SETUP_CLOBBER_CLASS_SETUP_CLOBBER:
                    setup_clobber_count++;
                    break;
                case VPM_SETUP_CLOBBER_CLASS_BOTH:
                    both_count++;
                    break;
                default:
                    other_count++;
                    break;
                }

                printk("%d,%d,%d,%d,%d,%d,%d,%s,%x,%x,%d\n",
                       (int)record->trial,
                       (int)record->qpu_a,
                       (int)record->qpu_b,
                       (int)record->slice_a,
                       (int)record->slice_b,
                       (int)record->row_a_match,
                       (int)record->row_b_match,
                       classification_name(record->classification),
                       record->errstat_before,
                       record->errstat_after,
                       (int)record->timeout);

                print_pair_diagnostic(record);

                if (record->timeout)
                    stop_after_timeout = 1;
            }
        }
    }

    int elapsed = timer_get_usec() - start;
    uint32_t conclusion_id = 0;
    if (valid_pairs != 0) {
        if (no_clobber_count == valid_pairs)
            conclusion_id = 1;
        else if (setup_clobber_count == valid_pairs)
            conclusion_id = 2;
        else
            conclusion_id = 3;
    }

    uint32_t runtime_allocations =
        vpm_setup_clobber_run_pair_runtime_allocations();
    uint32_t runtime_launches =
        vpm_setup_clobber_run_pair_runtime_launches();

    const char *status =
        (invalid_topology == 0 &&
         representative_pairs == 18u &&
         pairs_run == 18u &&
         valid_pairs == 18u &&
         qpu_mismatches == 0u &&
         timeouts == 0u &&
         errstat_relevant_changed_count == 0u &&
         runtime_allocations == 1u &&
         runtime_launches == pairs_run) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=vpm_setup_clobber status=%s representative_pairs=%d pairs_run=%d valid_pairs=%d setup_clobber_count=%d no_clobber_count=%d both_count=%d other_count=%d qpu_mismatches=%d timeouts=%d invalid_topology=%d errstat_relevant_changed=%d runtime_allocations=%d runtime_launches=%d conclusion_id=%d elapsed_usec=%d\n",
           status,
           (int)representative_pairs,
           (int)pairs_run,
           (int)valid_pairs,
           (int)setup_clobber_count,
           (int)no_clobber_count,
           (int)both_count,
           (int)other_count,
           (int)qpu_mismatches,
           (int)timeouts,
           (int)invalid_topology,
           (int)errstat_relevant_changed_count,
           (int)runtime_allocations,
           (int)runtime_launches,
           (int)conclusion_id,
           elapsed);

    vc4Free(program, row_a_dev);
    vc4Free(program, row_b_dev);
    vc4Free(program, qpu_a_report_dev);
    vc4Free(program, qpu_b_report_dev);
    vc4_program_destroy(program);
}
