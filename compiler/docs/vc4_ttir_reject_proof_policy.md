# VC4 TTIR Permanent Reject Proof Policy

## 1. Purpose

This document defines the proof standard for permanent TTIR and value-surface
rejects in the VC4 target profile. It is a Phase 1 specification artifact only.
It does not add a verifier, importer, lowering, test, runtime behavior, or
hardware fixture.

The policy prevents difficult but supportable features from being mislabeled as
impossible. A feature may be permanently rejected only after the project proves
that VC4 cannot preserve the required semantics, even through slow emulation, or
that the feature is outside the VC4 compute-kernel scope.

## 2. Temporary reject versus permanent reject

A temporary reject means the feature is not implemented, not specified, or not
yet safe to lower in the current staged compiler. It says nothing by itself
about hardware impossibility.

A permanent reject means the source semantics cannot be preserved on the VC4
compute target, even with slower expansion, scalarization, runtime help, or
other emulation. Permanent rejects require the proof template in Section 3.

Locked VC4Kernel deterministic rejects are target-contract facts. They do not
automatically prove source-level permanent impossibility, because the value
layer may be able to preserve semantics through a different composite or
emulated lowering path.

## 3. Required proof template

Every permanent reject proof must answer:

1. What source construct is rejected?
2. What exact source semantics are required?
3. Can it be represented in the value surface?
4. Can it be represented in VC4Kernel directly?
5. Which VC4 mechanisms were considered: ALU, predicates, rotate, composite reductions, TMU, VPM QPU read/write, VDR, VDW, SFU, barriers/semaphores, scalar loops, host/runtime support?
6. Could slow emulation preserve semantics?
7. If no, why exactly not?
8. Is the reject due to hardware impossibility, source semantic mismatch, or project compute-scope exclusion?
9. What diagnostic should the user see?
10. What tests prove the diagnostic?

The proof must cite the relevant taxonomy row in
`compiler/docs/vc4_value_ttir_feature_taxonomy.json`, the value-surface or TTIR
profile section, and any VC4Kernel locked-contract row it relies on.

## 4. Hardware mechanisms that must be considered

Before claiming permanent impossibility, the proof must consider at least:

- QPU add-pipe and mul-pipe ALU operations;
- scalar arithmetic, address, and control operations;
- predicates, mask algebra, select, any, and all;
- dynamic rotate and rotate-derived composites;
- composite reductions;
- TMU safe-offset inactive loads;
- VPM QPU read/write;
- VDR global-to-VPM movement;
- VDW VPM/register-fragment stores;
- pack/unpack and subword storage modes;
- f16 storage conversion plus f32 compute;
- approximate SFU under explicit policy;
- barriers and semaphores;
- scalar loops and multi-step lowering;
- host/runtime support where it would preserve source semantics and remain in
  project scope.

Considering a mechanism does not mean the mechanism is accepted for direct
VC4Kernel lowering. It means the proof must explain why it cannot preserve the
source semantics directly, compositionally, or by emulation.

## 5. Slow emulation requirement

Could slow emulation preserve semantics? Every permanent reject must answer
this directly.

Slow emulation may include scalarization, loops, multiple SIMD-16 fragments,
temporary VPM staging, value-level composite shuffles, explicit read/modify/write
where legal, or host/runtime involvement if that remains within the compute
target contract.

Performance is not a permanent-reject proof. A feature that is slow, awkward,
or unimplemented is a temporary reject unless the proof shows semantics cannot
be preserved.

## 6. Diagnostic requirements

Diagnostics for rejects must state:

- the rejected source construct or profile family;
- whether the reject is temporary, value-surface policy, VC4Kernel contract, or
  permanent;
- the required source semantics that were not accepted;
- the first unsupported boundary;
- any explicit policy that would make a related feature legal;
- whether future emulation or future hardware proof could change the result.

Diagnostics must not rely on fixture names, paths, public names, status strings,
or generated-output special cases.

## 7. Examples of current locked VC4Kernel deterministic rejects

Current examples:

- direct TTGIR/NVIDIA IR first -> value-surface policy reject, not hardware
  impossibility;
- producer ops inside VC4Kernel -> VC4Kernel contract reject;
- sparse VDW direct store -> VC4Kernel contract reject;
- arbitrary direct shuffle/permutation -> VC4Kernel direct reject, but
  value-level emulation remains future supportable;
- native f16 arithmetic -> VC4Kernel contract/hardware-format reject;
- exact/default SFU math -> policy reject.

These examples are deterministic rejects at a specific boundary. Except for
out-of-scope non-compute hardware, they should not be rewritten as permanent
TTIR source rejects without a separate proof under this policy.

## 8. Examples that are not permanent rejects yet

The following are not permanent rejects yet:

- atomics;
- strict IEEE f32 reduction via slow scalar/control path;
- arbitrary value-level permutations via composite/emulation;
- non-16 block sizes via splitting/looping;
- some memory ordering/cache hints if semantically droppable.

These features may be temporary rejects, supportable later, or policy-dependent
until a later phase proves semantic preservation or impossibility.

## 9. How future phases update reject proofs

Future phases may add or update permanent reject proofs only with matching
taxonomy updates, diagnostics, and tests. A proof update must preserve the
locked lower path:

```text
value surface -> vc4kernel -> ssavc4 -> scheduled vc4
```

If a future hardware-proven phase changes the VC4Kernel contract, the
VC4Kernel surface lock, support matrix, audits, and mixed-claim evidence must be
updated before value/TTIR docs depend on the new target capability.

If a temporary reject becomes supportable through emulation, the taxonomy should
move to `supported_emulated`, `supported_composite`,
`supported_with_policy_caveat`, or `supportable_later`, depending on the proof.
If a temporary reject becomes permanent, the proof template in Section 3 must
be completed before the classification changes.
