// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for warp_reduce_sum.
//
// This test validates a single-QPU 16-lane f32 reduction primitive. Each QPU
// processes logical vectors round-robin by vector index, loads one 16-lane
// vector through TMU0 direct memory lookup, masks inactive final-vector lanes,
// reduces the vector lanes, replicates the sum, and stores the valid prefix
// through VPM/VDW.
//
// Candidate/codegen execution remains disabled until the backend can emit the
// qasm and launcher bundle.

module attributes {
"vc4.hardware_run_test.name" = "warp_reduce_sum",
"vc4.hardware_run_test.kind" = "hardware-run-reference",
"vc4.hardware_run_test.reference_kernel" = "reference/warp_reduce_sum.qasm",
"vc4.hardware_run_test.launch_abi" = {
public_name = "warp_reduce_sum_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 5 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
],
builtins = [
{name = "qpu_id", kind = "qpu_num", materialization = "uniform_suffix", uniform_index = 3 : i32},
{name = "num_qpus", kind = "num_qpus", materialization = "uniform_suffix", uniform_index = 4 : i32}
],
work_distribution = {
logical_warp = "one QPU request",
lane = "ELEMENT_NUMBER",
vector_index = "qpu_id + chunk * num_qpus",
vector_width = 16 : i32,
tail_mask = "inactive final-vector lanes become zero before reduction",
store_tail = "dynamic_vdw_depth"
},
memory_paths = {
global_loads = "TMU0 direct memory lookup",
lane_reduction = "QPU rotate/add horizontal f32 reduction",
global_stores = "VPM staging plus VDW DMA store",
vpm_vdw_serialization = "global_mutex"
},
runtime_resource_policy = {
gpu_allocations_per_boot = 1 : i32,
code_copies_per_boot = 1 : i32,
repeated_launches = 13 : i32,
active_qpus = 12 : i32,
lane_width = 16 : i32
}
}
} {
}

