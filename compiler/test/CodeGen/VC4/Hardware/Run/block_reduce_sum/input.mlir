// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for block_reduce_sum.
//
// This test validates one cooperative CUDA-like block reduction wave at a time.
// Every logical warp in the block is queued together. Each warp computes a
// partial sum for one 16-value segment, writes the partial into a VPM row,
// synchronizes with the locked four-semaphore reusable barrier, and the leader
// reads the partial rows to compute and store the replicated block result.
//
// Candidate/codegen execution remains disabled until the backend can emit the
// qasm and launcher bundle.

module attributes {
"vc4.hardware_run_test.name" = "block_reduce_sum",
"vc4.hardware_run_test.kind" = "hardware-run-reference",
"vc4.hardware_run_test.reference_kernel" = "reference/block_reduce_sum.qasm",
"vc4.hardware_run_test.launch_abi" = {
public_name = "block_reduce_sum_launch",
tail_policy = "tail_safe",
uniform_words_per_qpu = 6 : i32,
args = [
{name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
{name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
{name = "values_per_block", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
],
builtins = [
{name = "logical_warp_id", kind = "warp_id", materialization = "uniform_suffix", uniform_index = 3 : i32},
{name = "warps_per_block", kind = "warps_per_block", materialization = "uniform_suffix", uniform_index = 4 : i32},
{name = "vpm_base_row", kind = "workgroup_memory_base", materialization = "uniform_suffix", uniform_index = 5 : i32}
],
work_distribution = {
device_model = "one_cuda_like_sm",
resident_blocks_per_wave = 1 : i32,
max_warps_per_block = 12 : i32,
lane = "ELEMENT_NUMBER",
partial_segment = "logical_warp_id * 16 + local_index",
shared_memory = "one VPM row per warp partial",
barriers = "four_semaphore_reusable_after_partial_write_and_after_leader_store",
result_shape = "one 16-lane replicated output vector per block"
},
memory_paths = {
global_loads = "TMU0 direct memory lookup",
workgroup_memory = "VPM horizontal 32-bit rows",
global_stores = "VPM staging plus VDW DMA store",
vpm_vdw_serialization = "global_mutex"
},
runtime_resource_policy = {
gpu_allocations_per_boot = 1 : i32,
code_copies_per_boot = 1 : i32,
repeated_launches = 6 : i32,
active_qpus = 12 : i32,
lane_width = 16 : i32,
semaphore_ids = "0,1,2,3",
multi_block_policy = "sequential_resident_waves"
}
}
} {
}

