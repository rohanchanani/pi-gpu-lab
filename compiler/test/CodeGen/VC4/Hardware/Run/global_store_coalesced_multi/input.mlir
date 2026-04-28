// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for global_store_coalesced_multi.
//
// This test isolates the generated global-store abstraction for VC4:
// vector values are built in QPU registers, staged through one VPM row per QPU,
// and written through VDW with dynamic DEPTH for tails. There are no TMU loads,
// floating-point operations, barriers, shared-memory reuse, or scatter stores.
//
// Candidate/codegen execution remains disabled until the backend can emit the
// qasm and launcher bundle.

module attributes {
"vc4.hardware_run_test.name" = "global_store_coalesced_multi",
"vc4.hardware_run_test.kind" = "hardware-run-reference",
"vc4.hardware_run_test.reference_kernel" = "reference/global_store_coalesced_multi.qasm",
"vc4.hardware_run_test.launch_abi" = {
public_name = "global_store_coalesced_multi_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 5 : i32,
args = [
{name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
{name = "case_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
],
builtins = [
{name = "qpu_id", kind = "qpu_num", materialization = "uniform_suffix", uniform_index = 3 : i32},
{name = "num_qpus", kind = "num_qpus", materialization = "uniform_suffix", uniform_index = 4 : i32}
],
work_distribution = {
logical_warp = "one QPU request",
lane = "ELEMENT_NUMBER",
base_element = "qpu_id * 16 + chunk * num_qpus * 16",
stride_elements = "num_qpus * 16",
store_tail = "dynamic_vdw_depth"
},
memory_paths = {
global_loads = "none",
global_stores = "VPM staging plus VDW DMA store",
vpm_vdw_serialization = "global_mutex"
},
runtime_resource_policy = {
gpu_allocations_per_boot = 1 : i32,
code_copies_per_boot = 1 : i32,
repeated_launches = 19 : i32,
active_qpus = 12 : i32,
lane_width = 16 : i32
}
}
} {
}

