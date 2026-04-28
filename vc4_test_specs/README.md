# VC4 automated minimal-friction bundle generation

This directory contains the committed inputs and small committed state markers for `pro_scripts/vc4_bundle_autorun.py`.

The driver is deterministic. ChatGPT generates or repairs bundle files. Codex performs only the generated mechanical prompt. Python owns staging, validation, retries, test execution, quarantine, state markers, and commits.

## Required files

```text
vc4_test_specs/
  god_prompt.md
  manifest.tsv
  specs/
    <number>_<test-name>.txt
  state/
    passed/
      <test-name>.json
    incomplete/
      <test-name>.json
```

`state/` is created by the driver. Passed and incomplete marker JSON files are committed so the driver is resumable.

The driver also writes untracked logs and archives under:

```text
.vc4_auto/
```

The driver adds `.vc4_auto/` to `.git/info/exclude` automatically.

## Existing project files you should provide as context

Put the full cold-chat god prompt in:

```text
vc4_test_specs/god_prompt.md
```

Use `--extra-context-file` for large files that should be pasted only into fresh-tab prompts. The current most useful one is:

```text
compiler/dialect.txt
```

Example:

```bash
python3 pro_scripts/vc4_bundle_autorun.py \
  --repo . \
  --extra-context-file compiler/dialect.txt
```

The extra context is included when the driver opens a new tab with `god_prompt.md + context + spec`. It is not repeated for current-tab follow-up specs or repair prompts because the current tab should already contain it.

## Manifest format

`manifest.tsv` is tab-separated:

```text
TEST_NAME<TAB>SPEC_PATH
```

Rules:

- Blank lines are ignored.
- Lines beginning with `#` are ignored.
- `TEST_NAME` must use letters, digits, underscore, or hyphen.
- `SPEC_PATH` is relative to the repo root unless absolute.
- The spec file must contain a line exactly matching `TEST_NAME: <name>`.

Example:

```text
matmul_blocked	vc4_test_specs/specs/001_matmul_blocked.txt
vpm_setup_clobber	vc4_test_specs/specs/002_vpm_setup_clobber.txt
```

## Locked test specification format

Each spec file is plain text. The driver only validates `TEST_NAME`, but the model-facing locked format is:

```text
TEST_NAME: <name>
TEST_KIND: hardware-run
ONE_SENTENCE_PURPOSE: <one sentence>

WHY_THIS_TEST_NOW:
<why this is the next incremental hardware/codegen concept>

SEMANTICS:
<kernel semantics, mathematical operation, memory model, launch shape>

PUBLIC_LAUNCH_API:
<semantic public API expected in <test-name>_launch.h>

RUNTIME_SETUP_REQUIREMENTS:
<max sizes, one-allocation setup, resident blocks, active QPUs, VPM/semaphore resources>

QPU_KERNEL_SHAPE:
<logical warp/lane mapping, uniforms, TMU/VPM/VDW paths, barrier use if any>

TEST_CASES:
<deterministic cases and edge cases>

EXPECTED_OUTPUT:
<diagnostic lines and final VC4_TEST_RESULT fields>

EXPECTED_JSON_ORACLE:
<stable fields expected.json should require; no timing>

NON_GOALS:
<what this test intentionally does not prove>

SPECIAL_NOTES:
<any hazards, hardware constraints, or known-good style references>
```

Do not put the god prompt in individual specs. The driver provides the god prompt only when opening a fresh tab.

## Resume behavior

The driver skips tests already marked as passed or incomplete:

```text
vc4_test_specs/state/passed/<test>.json
vc4_test_specs/state/incomplete/<test>.json
```

To retry an incomplete or passed test with modifications:

```bash
python3 pro_scripts/vc4_bundle_autorun.py --reset-test <test-name>
```

or manually remove the relevant state marker and commit that removal.

The driver starts the first unprocessed test in a fresh ChatGPT tab with `god_prompt.md + extra context + spec`. After a pass, the next test uses the current tab with only the next spec. After an incomplete test, the next test starts a fresh tab with `god_prompt.md + extra context + spec`.

## Timeouts

Defaults:

```text
Chat invocation:        20 minutes
Codex mechanical step:  10 minutes
vc4-opt:                 2 minutes
check-vc4:               2 minutes
hardware runner:         2 minutes
```

Override with command-line flags if a supervised run shows a specific test needs more time.

## Git behavior

After a passed test:

```text
git commit -m "passed <test-name>"
```

After an incomplete test:

```text
git commit -m "incomplete <test-name>"
```

The driver never pushes.

Before running unattended, commit this setup and any specs so the repo starts clean.
