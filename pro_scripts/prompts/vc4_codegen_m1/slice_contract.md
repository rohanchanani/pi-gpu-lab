# VC4 Codegen Milestone 1 Slice Contract

A slice is the smallest unit of GPT-driven compiler work. Each slice has an explicit ID, prerequisites, context profile, allowed paths, forbidden paths, gates, retry limits, and Codex policy in `pro_scripts/vc4_codegen_m1_worklist.json`.

## One-slice rule

Every GPT Pro implementation prompt is for exactly one slice. GPT Pro must not implement future slices opportunistically.

Examples:

- In `m1-03-minimal-thrend-qasm`, emit only enough qasm to assemble `thrend; nop; nop`.
- In `m1-04-qpu-bundle-basic`, do not implement `vc4.qpu.ldi`, `vc4.qpu.sema`, or branch handling.
- In `m1-07-launch-abi-model-header`, do not implement full launcher C uniform packing.

## Patch scope rule

A slice patch may touch only the paths listed in that slice's `allowed_paths` and may never touch the slice's `forbidden_paths`.

The autorun script must reject patches that violate the path policy before applying them.

## Test rule

Substantive compiler behavior must be protected by a local deterministic test whenever possible. Prefer local non-hardware tests under:

```text
compiler/test/CodeGen/VC4/Emit/**
```

Hardware gates are required only for hardware slices, and once a slice declares a hardware gate, the gate cannot be accepted by a local-only substitute.

## Verifier boundary rule

Milestone 1 codegen consumes verifier-clean final scheduled QPU functions. It must not silently repair invalid scheduled IR.

The required scheduled verifier pipeline is:

```text
--vc4-verify-emit-contract
--vc4-verify-scheduled-hardware-rules
--vc4-verify-scheduled-adjacent-hazards
--vc4-verify-scheduled-io-spacing
--vc4-verify-scheduled-peripheral-accesses
```

If the input violates the Milestone 1 boundary, the tool should fail cleanly with a useful diagnostic rather than broaden the accepted input language.

## Reference immutability rule

Do not change reference bundles. Do not update `expected.json` to make candidate output pass. Do not update `catalog.json` as part of codegen implementation.

## Auto-commit rule

After a slice's gates pass, the autorun script may create a commit using the slice's configured `commit_message`. A later slice depends on the committed state of its prerequisites.
