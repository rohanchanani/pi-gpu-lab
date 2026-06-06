# VC4Kernel Mixed Acceptance Policy

P8.5 changes final hardware acceptance from replaying every historical isolated
fixture to running a smaller mixed suite that exercises P1-P8 interactions. The
isolated fixtures remain in tree as phase-local proofs and triage tools; they
are not weakened, deleted, or removed from explicit phase prompts.

## Routine Acceptance

After P8.5, routine final acceptance uses:

- the mixed acceptance manifest coverage checker;
- the mixed acceptance hardware runner once the suite is locked;
- Surface v2 audits and support-matrix checks;
- lit coverage and `check-vc4`.

Routine final acceptance does not run every isolated historical hardware
fixture by default. A phase prompt may still request isolated fixtures, and
triage should run the directly related isolated band when a mixed fixture fails.

## Manifest And Runner

The mixed suite is described by:

`compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json`

The manifest lists the required mixed fixtures, required P1-P8 feature coverage,
deterministic-reject coverage, and future P9-P13 extension points. Planned mode
allows fixture files to be absent while the suite is being added. Lock mode
requires every fixture input and expected result file to exist and verifies that
VC4Kernel mixed fixtures do not contain source-authored SSAVC4, scheduled VC4,
or producer dialects.

`run_mixed_acceptance.sh` reads the manifest and runs implemented fixtures with
the existing strict candidate hardware runners. It supports `--list` and
repeated `--fixture` arguments for triage. It creates a fresh
`VC4_CODEGEN_STATE_ROOT` by default, writes logs under that root, and prints
every `VC4_TEST_RESULT` line.

## Failure Triage

When a mixed fixture fails:

1. Keep the mixed fixture and oracle intact.
2. Run the isolated fixture band for the failed feature group.
3. If isolated fixtures pass, build a smaller interaction ladder.
4. Fix the compiler, lower half, runtime, or fixture bug and rerun the mixed
   fixture.

Deterministic rejects remain lit or audit tests. They are not hardware fixtures.

## Future Phases

P9-P13 must add mixed coverage for new accepted features instead of restoring
the old every-isolated-fixture final matrix. Future mixed additions should keep
the same strict requirements: CPU oracle, sentinels, expected PASS metadata, no
fixture-specific compiler special cases, and real hardware execution.
