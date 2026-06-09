This addendum applies to all future VC4 compiler work. It is intentionally strict because the compiler is now moving from smoke-proven vertical slices into reusable infrastructure.

### A. No temporary architecture

Do not land code that is known to be a temporary, brittle, stringly typed, fixture-specific, or workaround-shaped foundation.

Examples of forbidden foundations:

- semantic decisions based on printed IR/type/attribute substrings;
- fixture-name, test-path, public-name, candidate-name, or status-string special cases;
- Python semantic import paths for accepted TTIR lowering after the C++ importer is locked;
- output IR produced by unlinking/leaking old operations instead of using proper MLIR ownership;
- bypassing an intended layer because the proper layer is difficult;
- accepting “we will clean this up later” for code that future phases must build on.

If the correct long-term design is clear, implement that design even if it is harder. If the correct long-term design is ambiguous, stop with `FAILURE` or `BLOCKED` and name the exact design choices that require user input. Do not force through a short-term design.

### B. Semantic classification must be structural

Compiler semantics must come from typed APIs, operation classes, exact operation names, exact attributes, SSA use-def structure, verified dialect contracts, and documented hardware facts.

Allowed string uses:

- exact operation-name constants when generic MLIR operation handling is deliberate;
- exact attribute-name lookup followed by typed/exact interpretation;
- diagnostics, reports, and debug/provenance output;
- source-name metadata that does not decide semantic roles.

Forbidden semantic string uses:

- parsing printed types to decide lowering behavior;
- parsing printed operations or attributes to decide lowering behavior;
- permissive substring matching such as “contains x” or “contains 0” for semantic decisions;
- using source argument names to decide pointer direction, tail-bound role, memory role, or type legality;
- using comments or documentation as a substitute for verifier/conversion logic and tests.

Every new frontend/value feature must have audits that enforce this boundary where practical.

### C. Use proper MLIR ownership and output construction

Passes must not rely on unlinking, leaking, or leaving old operations in an invalid ownership state.

For importer/conversion passes that replace one IR layer with another, prefer:

- analyze input without mutating it;
- build a clean output representation;
- verify the output boundary;
- commit only after the full conversion succeeds;
- preserve atomic failure behavior;
- destroy or replace old operations using standard MLIR ownership mechanisms.

Do not use `Operation::remove()` or similar unlinking as a long-term solution to avoid erasure/destruction issues. If proper erasure exposes an assertion or ownership bug, stop and fix the ownership/modeling problem.

### D. Preserve layer boundaries under pressure

The intended stack remains:

```text
TTIR
  -> standard VC4 value layer
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> hardware
```

Do not collapse layers to make a phase pass.

Examples:

- TTIR must not lower directly to `vc4kernel`.
- Value IR must not lower directly to `ssavc4` or scheduled `vc4`.
- Verified `vc4kernel` must not contain producer dialect operations.
- `vc4value` must remain tiny launch/policy plumbing.
- Default builds must not depend on optional Triton C++ dependencies.
- The optional Triton C++ dependency closure must remain quarantined behind the dedicated adapter target.

### E. Package quality bar

Every nontrivial phase package should have the following shape unless the user explicitly asks otherwise:

1. Read-only inventory/design lock.
2. Narrow implementation prompt(s) with explicit scope.
3. Static verifier/lit/audit coverage.
4. Hardware isolation fixtures for new executable semantics.
5. Mixed acceptance fixtures or mixed regression updates.
6. Claim/coverage/doc/audit lock.
7. Final acceptance that either unlocks the next phase or reports exact blockers.

Do not leave a phase with vague “future cleanup” items if they affect the foundation for the next phase. Either fix them in the phase or mark the phase blocked with a concrete blocker.

### F. Hardware and regression standard

If a phase changes executable semantics or any lowering used by executable semantics, it must run real hardware unless the prompt is explicitly read-only/static.

Hardware acceptance must preserve:

- real source input;
- all intermediate IR/artifacts;
- CPU oracle;
- sentinel checks;
- mismatch/launch-failure accounting;
- active QPU coverage appropriate to the feature, normally 12 active QPUs for mixed acceptance;
- nonzero output/work hash where relevant.

Do not reduce fixture coverage, simplify the kernel, or weaken expected results to make hardware pass.

### G. Debugging expectations

Debug aggressively and scientifically.

When a mixed fixture fails:

- preserve all artifacts;
- inspect every layer of the pipeline;
- run relevant isolation fixtures;
- remove/add features to isolate the interaction;
- instrument the current or lower layer if needed;
- consider lower-half bugs as possible;
- stop and flag true lower-half regressions rather than hiding them.

Do not merely stare at code or make speculative patches without testing the hypothesis.

### H. Copyable terminal commands

When writing commands meant to be pasted directly into a terminal, do not include shell comment lines beginning with `#`.

Explanations belong outside the command block. Copyable shell blocks should contain executable shell only.

### I. Failure is acceptable; false success is not

Use `FAILURE` or `BLOCKED` when:

- the long-term design is unclear;
- a correct implementation would require a larger prerequisite;
- a required command fails;
- a verifier/audit/hardware result fails;
- passing would require a shortcut;
- a source foundation issue is found during a later phase.

Final responses must name exact blockers and next actions. Do not claim success with hidden caveats that make the next phase unsafe.
