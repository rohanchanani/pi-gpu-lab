#include "rpi.h"
#include "vpm_setup_clobber_launch.h"

#define ERRSTAT_RELEVANT_MASK 0x0000efffu /* exclude VCD idle bit 12 */
#define VPM_SETUP_CLOBBER_TRIALS_PER_PAIR 3u
#define VPM_SETUP_CLOBBER_MAX_RECORDS 32u

struct qpu_pair {
uint32_t a;
uint32_t b;
};

static struct vpm_setup_clobber_record records[VPM_SETUP_CLOBBER_MAX_RECORDS];

static uint32_t errstat_relevant_changed(uint32_t before, uint32_t after)
{
return (((before ^ after) & ERRSTAT_RELEVANT_MASK) != 0);
}

static void print_topology_json(
const struct vpm_setup_clobber_topology *topology)
{
printk("vc4_vpm_setup_clobber_topology.json:\n");
printk("{\n");
printk(" ident1: %x,\n", topology->ident1);
printk(" vpmsz_field: %d,\n", (int)topology->vpmsz_field);
printk(" vpm_kib: %d,\n", (int)topology->vpm_kib);
printk(" qpus_per_slice: %d,\n", (int)topology->qpus_per_slice);
printk(" num_slices: %d,\n", (int)topology->num_slices);
printk(" num_semaphores: %d,\n", (int)topology->num_semaphores);
printk(" num_qpus: %d,\n", (int)topology->num_qpus);
printk(" vpmbase_written: %d,\n", (int)topology->vpmbase_written);
printk(" vpmbase_readback: %x\n", topology->vpmbase_readback);
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
vpm_setup_clobber_classification_name(record->classification),
(int)record->qpu_mismatches,
(int)record->timeout,
record->errstat_before,
record->errstat_after,
record->row_a[0],
record->row_b[0],
0xe5000000u | ((record->qpu_a & 0x0fu) << 8));
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
const struct vpm_setup_clobber_topology *topology,
struct qpu_pair pairs[VPM_SETUP_CLOBBER_MAX_RECORDS])
{
uint32_t count = 0;
uint32_t qps = topology->qpus_per_slice;
uint32_t nslices = topology->num_slices;
uint32_t nq = topology->num_qpus;

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

void notmain(void)
{
struct vc4_runtime rt;
struct vpm_setup_clobber_state state;
struct qpu_pair pairs[VPM_SETUP_CLOBBER_MAX_RECORDS];

if (vc4_runtime_init(&rt) < 0)
    panic("Failed to initialize VC4 runtime");

printk("Running VC4 vpm_setup_clobber reference bundle...\n");

if (vpm_setup_clobber_prepare(&rt, &state) < 0)
    panic("vpm_setup_clobber runtime setup failed");

print_topology_json(&state.topology);

uint32_t invalid_topology = 0;
if (state.topology.qpus_per_slice != 4u)
    invalid_topology = 1;
if (state.topology.num_slices < 3u)
    invalid_topology = 1;
if (state.topology.num_qpus < 12u)
    invalid_topology = 1;
if (state.topology.num_semaphores < 16u)
    invalid_topology = 1;
if ((state.topology.vpmbase_readback & 0x1fu) !=
    VPM_SETUP_CLOBBER_VPM_URSV_4K)
    invalid_topology = 1;

uint32_t pair_count = build_representative_pairs(&state.topology, pairs);
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
            int rc = vpm_setup_clobber_run_pair(&state,
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
                err_changed == 0) {
                valid_pairs++;
            }

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
                   vpm_setup_clobber_classification_name(record->classification),
                   record->errstat_before,
                   record->errstat_after,
                   (int)record->timeout);

            print_pair_diagnostic(record);

            if (record->timeout)
                stop_after_timeout = 1;
        }
    }
}

int end = timer_get_usec();
int elapsed = end - start;

uint32_t conclusion_id = 0;
if (valid_pairs != 0) {
    if (no_clobber_count == valid_pairs)
        conclusion_id = 1;
    else if (setup_clobber_count == valid_pairs)
        conclusion_id = 2;
    else
        conclusion_id = 3;
}

const char *status =
    (invalid_topology == 0 &&
     representative_pairs == 18u &&
     pairs_run == 18u &&
     valid_pairs == 18u &&
     qpu_mismatches == 0u &&
     timeouts == 0u &&
     errstat_relevant_changed_count == 0u &&
     state.runtime_allocations == 1u &&
     state.runtime_launches == pairs_run) ? "PASS" : "FAIL";

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
       (int)state.runtime_allocations,
       (int)state.runtime_launches,
       (int)conclusion_id,
       elapsed);

/*
 * Keep the single GPU allocation live until the hardware runner
 * power-cycles the Pi. This avoids introducing an alloc/free loop into the
 * litmus. The shutdown API exists for standalone callers.
 */
vc4_runtime_shutdown(&rt);

}

