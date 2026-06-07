# VC4Kernel Mixed Acceptance Policy

Mixed hardware acceptance is the routine final gate for Surface v2 and for producer-layer work that lowers into VC4Kernel. A mixed fixture is accepted only when it is listed in the mixed acceptance manifest, generated from source, run through the strict hardware runner, and checked by the result oracle.

## Fixture Requirements

Each mixed fixture must have:

- A VC4Kernel or SSAVC4 input that respects the final stack boundary.
- A host harness and CPU/reference oracle where the fixture writes memory.
- Sentinel or guard checks for inactive lanes, preserve paths, and launch/resource failures where relevant.
- Expected result metadata that is specific enough for the checker to reject mismatches.
- A mixed fixture claim entry for every `saw_*`, `no_*`, or equivalent result field.

Verified VC4Kernel hardware fixtures do not contain producer dialect operations. Mixed SSAVC4 lower-half fixtures may test lower-half behavior, but they do not create a direct VC4Kernel-to-VC4 route.

## Claim Kinds

Claims are listed in:

```
compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_fixture_claims.json
```

Allowed claim kinds:

- `CHECKED_OUTPUT`: the feature affects a checked output buffer.
- `CHECKED_AUDIT`: the feature affects a checked audit value, checksum, guard, or metadata field.
- `PHASE_GUARD`: the claim is a suite-level statement backed by named guard fixtures required in the same targeted/final run.

The audit script rejects claims without evidence, dead feature operations for checked-output claims, broad "full" wording without enumerated claims, and phase guards that do not name required fixtures.

## P12 Dynamic Selector Contract

P12 mixed acceptance covers dynamic VPM/VDR/VDW row, word-X, and subword selector behavior without merging their meanings. Dynamic subword selector is a byte/halfword selector. It is not dynamic width, dynamic orientation, or dynamic subword mode.

The policy distinguishes:

- VDR raw carrier placement.
- QPU-normalized subword readback.
- VDW memory output and preserve behavior.
- Mixed dataflow that makes selector behavior affect checked output or checked audit data.

The ping-pong copy-path fixture does not claim QPU VPM readback unless the readback contributes to checked output. Ping-ponged QPU readback is covered by its checked fixture and by the claim contract. Horizontal-only x=0 compute fixtures state that scope directly; nonzero X values are covered by named guard fixtures or final mixed fixtures.

## Final Mixed Run

The final mixed suite must include the accepted P8.5, P9, P10, P11, and P12 mixed fixtures and any required guard fixtures referenced by claim entries. Passing final mixed acceptance requires `VC4_TEST_RESULT` status `PASS` and zero mismatch, sentinel, and launch-failure fields where those fields are emitted.

Hardware artifacts under `.vc4_auto` are generated outputs and are not source-of-truth inputs for the claim contract.

## Isolated Fixture Band

Representative isolated fixtures remain part of the proof record. Routine final acceptance does not run every isolated hardware fixture; it runs the final mixed suite and the named guard fixtures required by claim entries. Run the isolated fixture band when a source or harness change touches the directly covered behavior. P9-P13 mixed coverage is present in the final manifest.
