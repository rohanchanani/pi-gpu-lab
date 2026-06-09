# VC4 Value/Triton Feature Taxonomy Schema

## 1. Purpose

This document defines the schema used by
`compiler/docs/vc4_value_ttir_feature_taxonomy.json`.

The taxonomy is a Phase 1 specification artifact only. It classifies future
producer-facing value and TTIR features before implementation. It does not add
`vc4value`, does not add a verifier, does not import TTIR, does not lower value
IR, and does not claim executable Triton support.

The taxonomy exists so later phases can separate:

- features that map to the locked VC4Kernel Surface v2;
- features that are expressible as canonical composites;
- features that need emulation or policy caveats;
- features that are only staged out because no lowering exists yet;
- target forms rejected by the locked VC4Kernel contract;
- features that require a permanent hardware/semantic impossibility proof.

## 2. Stack boundary

The classified stack is:

```text
Triton / real TTIR
  -> standard MLIR value surface
       func + tiny vc4value + vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

The locked lower path is value layer -> vc4kernel -> ssavc4 -> scheduled vc4
-> artifacts/runtime/hardware. VC4Kernel never lowers directly to scheduled
VC4. VC4Tile is retired and must not be revived.

VC4Kernel may use `vector<16xT>` carrier types for target fragments, but
verified VC4Kernel hardware fixtures must not contain vector dialect operations
or other producer dialect operations.

The Phase 3.5 value abstraction policy distinguishes surface-admissible fixed
vectors from Phase 5 V1 lowerable `vector<16>` fragment-normal forms. Taxonomy
rows must not describe `vector<16xT>` as the global value-layer vector limit.

`READY_FOR_TRITON` remains `NO` during Phase 1. TTIR/Triton rows in the taxonomy
are planning classifications, not implementation claims.

## 3. Status definitions

`supported_native`
: The feature maps directly to accepted VC4Kernel semantics and does not require
  a source-level caveat, though implementation may be staged.

`supported_composite`
: The feature is expressed using a canonical composition of accepted VC4Kernel
  primitives, not a new first-class VC4Kernel op.

`supported_emulated`
: The feature can preserve semantics through slower expansion or multi-step
  lowering; performance may be poor.

`supported_with_policy_caveat`
: The feature lowers only when the source or compiler policy explicitly accepts
  finite, approximate, storage-conversion, or other non-default semantics.

`supportable_later`
: Hardware/VC4Kernel support appears sufficient, but this phase intentionally
  does not design the full lowering yet.

`temporary_reject_not_implemented`
: The feature is not accepted in the current staged implementation, but no
  permanent hardware/semantic impossibility has been proven.

`deterministic_reject_value_surface_policy`
: The value surface refuses the feature as an initial source-boundary policy,
  even if a later phase may support it.

`deterministic_reject_vc4kernel_contract`
: The locked VC4Kernel surface rejects the corresponding target form.

`permanent_reject_hardware_or_semantic_impossible`
: VC4 cannot preserve the required semantics even through slow emulation;
  requires a proof under `compiler/docs/vc4_ttir_reject_proof_policy.md` once
  that file exists.

`out_of_scope_non_compute_hardware`
: Fixed-function graphics/texture/tile-buffer/display features outside the VC4
  compute-kernel target.

`internal_only`
: Resource/manifest/runtime/audit/artifact contract, not a source-program
  feature.

`frontend_specialization`
: The source construct is resolved by the Triton frontend or TTIR generation
  before runtime value control flow exists. No runtime value control-flow import
  is required when the emitted TTIR has no corresponding semantic operation.

`phase_inventory_target`
: A future phase must inspect real emitted TTIR forms before support or reject
  classification is locked.

`lowerable_if_matches_value_cf_subset`
: A TTIR control-flow form may lower only when structural analysis proves it
  matches the locked Phase 8 value-cf subset. Non-matching forms remain staged
  or reject with the first unsupported boundary named.

## 4. Feature row schema

Each `feature_families` row is an object with these fields:

- `feature_id`: stable lowercase dot-separated identifier.
- `family`: coarse family name for grouping.
- `source_layers`: array drawn from `triton_source`, `ttir`,
  `value_surface`, `vc4kernel_dependency`, and `hardware`.
- `source_constructs`: source, value, or TTIR constructs covered by the row.
- `value_surface_form`: intended value-layer representation.
- `required_vc4kernel_features`: locked VC4Kernel features needed downstream.
- `hardware_mechanisms`: VC4 mechanisms relevant to the row.
- `classification`: one status from Section 3.
- `first_planned_phase`: first later phase expected to own detailed design or
  implementation.
- `math_policy`: math semantic policy or `null`.
- `memory_policy`: memory semantic policy or `null`.
- `mask_policy`: mask semantic policy or `null`.
- `subword_policy`: subword/storage policy or `null`.
- `diagnostic_policy`: diagnostic behavior required before implementation.
- `permanent_reject_proof_required`: true only for permanent rejects.
- `notes`: human-readable rationale and locked-contract reminders.

Feature rows are not test manifests and do not imply a lowering exists.

## 5. Phase target meanings

`phase2`
: Initial value-surface documentation/spec work.

`phase3`
: First narrow handwritten value lowering slices.

`phase4`
: Core vector, mask, elementwise, and simple memory lowering expansion.

`phase5`
: Reductions, math policy, f16 storage, subword, and shuffle policy expansion.

`phase6`
: Tiling, VPM/VDR/VDW planning, contracts, and GEMV/GEMM-oriented planning.

`phase6_inventory_only`
: Inventory or profile-only row that must not be implemented in Phase 6.

`phase7`
: Real TTIR ingestion and target-profile hookup after handwritten value paths
  are proven.

`future_hardware_proof`
: Requires a future hardware-proven phase before a rejected target contract can
  change.

`never_compute_target`
: Outside the VC4 compute-kernel target.

## 6. Temporary reject versus permanent reject

A temporary reject means the project has not implemented or specified a safe
lowering yet. It does not say the feature is impossible.

A permanent reject requires proof that VC4 cannot preserve the source semantics,
even with a slow expansion or emulation path. Phase 1b should avoid permanent
rejects unless the proof obligation is already clear. Once
`compiler/docs/vc4_ttir_reject_proof_policy.md` exists, any permanent reject row
must cite that policy and the proof artifact it requires.

Locked VC4Kernel rejects are different from permanent value/TTIR rejects. For
example, sparse VDW stores and arbitrary direct VC4Kernel permutations are
deterministic VC4Kernel-contract rejects, while a future value planner may still
choose an emulated non-VC4Kernel-direct strategy if semantics can be preserved.

## 7. Policy fields

`math_policy` records finite-only, approximate SFU, storage-conversion, exact
basic arithmetic, or temporary exact-complex-math restrictions. Exact/default
math must not silently lower to approximate SFU.

`memory_policy` records memref/vector memory semantics, contiguous/tail
handling, dense rectangular stores, cache hint handling, atomics staging, or
out-of-scope memory forms.

Vector rows should distinguish `surface_verifier_behavior` from
`phase5_lowering_behavior` once those fields are present. Fixed rank-1
non-16 vectors are staged for split/tail lowering, fixed rank-2 vectors are
staged for tile/contract planning, and scalable vectors remain deterministic
initial rejects.

`mask_policy` records full, empty, tail, rectangular, sparse-compute,
sparse-store, and unknown-mask behavior. Sparse VDW stores remain a locked
VC4Kernel reject.

`subword_policy` records static packed modes, byte/halfword selectors, f16
storage conversion, and the boundary that dynamic selector does not mean dynamic
width, subword mode, orientation, or layout.

## 8. How later phases update this taxonomy

Later phases may update a row only when they also update the matching spec,
tests, audits, or proof artifacts. Updates must preserve these invariants:

- no direct VC4KernelToVC4 path;
- no VC4Tile resurrection;
- no producer dialect operations inside verified VC4Kernel;
- no affirmative Triton-readiness line until real TTIR ingestion is implemented
  and proven;
- no permanent reject without semantic or hardware impossibility proof;
- no executable value/Triton support claim from documentation alone;
- no weakening of locked VC4Kernel verifier, audit, fixture, or mixed-claim
  contracts.

If a later phase proves hardware support for a currently rejected VC4Kernel
target form, it must update the VC4Kernel lock artifacts first; value/TTIR docs
cannot override the locked surface.

## 9. Non-goals and hard rules

This taxonomy does not:

- add a `vc4value` dialect or operations;
- add a value-surface verifier pass;
- add value-to-VC4Kernel conversion;
- add TTIR/Triton ingestion;
- add Triton-dependent tests;
- run hardware;
- modify VC4Kernel, SSAVC4, scheduled VC4, runtime, or fixture semantics.

Phase 1 docs may reference the locked VC4Kernel Surface v2, but they must not
weaken it or claim support beyond it.
