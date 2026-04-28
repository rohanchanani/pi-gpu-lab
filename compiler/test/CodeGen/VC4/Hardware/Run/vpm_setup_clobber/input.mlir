// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Exploratory final-stage metadata for vpm_setup_clobber.
//
// This hardware-run test intentionally exercises direct QPU hardware features
// that are not yet represented cleanly by the final-stage vc4 dialect as a
// scheduled sink body: physical QPU reservations, register-materialized
// QPU_NUMBER/ELEMENT_NUMBER, intentionally unprotected VPM setup writes,
// mutex-protected VPM readback/writeback, VDW stores, and QPU semaphore
// ordering.
//
// The trusted handwritten reference qasm is therefore the hardware source of
// truth for this VPM setup-state litmus. Candidate/codegen execution remains
// disabled until those operations can be represented and emitted faithfully.

module attributes {
"vc4.hardware_run_test.name" = "vpm_setup_clobber",
"vc4.hardware_run_test.kind" = "hardware-run-reference",
"vc4.hardware_run_test.reference_kernel" = "reference/vpm_setup_clobber.qasm",
"vc4.hardware_run_test.launch_abi" = {
public_name = "vpm_setup_clobber",
tail_policy = "fixed_topology_litmus",
uniform_words_per_qpu = 9 : i32,
args = [
{name = "qpu_a", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32},
{name = "qpu_b", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
{name = "trial", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
{name = "row_a_out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 5 : i32},
{name = "row_b_out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 6 : i32},
{name = "qpu_a_report", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 7 : i32},
{name = "qpu_b_report", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 8 : i32}
],
builtins = [
{name = "qpu_num", kind = "qpu_register", materialization = "register"},
{name = "elem_num", kind = "qpu_register", materialization = "register"},
{name = "mutex", kind = "qpu_io", materialization = "register"},
{name = "semaphore", kind = "qpu_signal", materialization = "instruction"},
{name = "vpm_vdw", kind = "qpu_io", materialization = "register"}
],
physical_uniform_stream = [
{index = 0 : i32, name = "qpu_a"},
{index = 1 : i32, name = "qpu_b"},
{index = 2 : i32, name = "trial"},
{index = 3 : i32, name = "clobber_row_a"},
{index = 4 : i32, name = "clobber_row_b"},
{index = 5 : i32, name = "row_a_out"},
{index = 6 : i32, name = "row_b_out"},
{index = 7 : i32, name = "qpu_a_report"},
{index = 8 : i32, name = "qpu_b_report"}
],
representative_pairs_12_qpu = [
{a = 0 : i32, b = 1 : i32},
{a = 1 : i32, b = 0 : i32},
{a = 0 : i32, b = 4 : i32},
{a = 4 : i32, b = 0 : i32},
{a = 0 : i32, b = 8 : i32},
{a = 8 : i32, b = 0 : i32}
],
trials_per_representative_pair = 3 : i32,
clobber_window = {
protected_by_mutex = false,
sequence = "A writes VPM setup row A; B writes VPM setup row B; A writes VPM_WRITE without reprogramming setup"
},
conservative_phases = {
clearing = "mutex-protected VPM writes",
readback = "mutex-protected VPM reads and VDW stores",
final_thread_end = "nop; thrend plus two nop delay slots"
}
}
} {
}

