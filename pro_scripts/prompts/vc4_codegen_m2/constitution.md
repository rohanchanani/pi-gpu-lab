# VC4 Codegen M2 Constitution

You are working on Milestone 2 of the VC4 code generator. M2 finishes the backend for already-scheduled `vc4` QPU kernels. It does not lower MLIR `gpu` dialect and it does not lower structured VC4 SSA into scheduled QPU instructions.

The backend must treat a single kernel as a one-element program, not as a special artifact path. Prefer durable contracts over one-off fixes. Do not mutate reference/oracle files for existing fixtures. New fixtures may add their own reference/oracle files only in slices that explicitly allow them.

CUDA-likeness means public generated launch APIs take a program handle, grid/block geometry, device pointers for buffers, and scalar values by value. The generated launch code packs uniforms and enqueues hardware; user/test harness code is responsible for explicit allocation and copies through `vc4Malloc`, `vc4Memcpy*`, and `vc4Free`.

When a verification fails, fix the narrow contract violation. Do not weaken the verifier or update expected JSON to match broken output. If a feature is outside M2, reject it with a deterministic diagnostic and document the deferral.
