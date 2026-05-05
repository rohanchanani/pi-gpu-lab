# VC4 Codegen Milestone 1 GPT Output Contract

GPT Pro must return exactly the files requested by the active prompt in GPTWEB file blocks.  For implementation/failure-fix prompts that means exactly:

1. `response.json`
2. `changes.patch`

Do not include prose outside GPTWEB file blocks.

## Required file block format

Use this exact outer form:

```text
BEGIN_GPTWEB_FILE path=response.json
{ ... valid JSON ... }
END_GPTWEB_FILE

BEGIN_GPTWEB_FILE path=changes.patch encoding=git-patch-lines
R|diff --git a/path b/path
R|--- a/path
R|+++ b/path
R|@@ ...
C| unchanged context line payload
D|deleted line payload without the leading minus
A|added line payload without the leading plus
END_GPTWEB_FILE
```

`changes.patch` must use `encoding=git-patch-lines`.  In that encoding each physical output line starts with one of:

- `R|` for raw diff metadata lines such as `diff --git`, `index`, `---`, `+++`, `@@`, `new file mode`, `deleted file mode`.
- `C|` for context lines; the driver decodes this to a leading space.
- `D|` for deletion lines; the driver decodes this to a leading `-`.
- `A|` for addition lines; the driver decodes this to a leading `+`.

Inside the payload, HTML-sensitive characters may be entity-shielded.  The driver decodes entities before writing the patch.  Do not use Markdown fences inside GPTWEB file blocks.

## `response.json` schema

Implementation and failure-fix responses must be a JSON object with these keys:

```json
{
  "summary": "one-sentence patch summary",
  "diagnosis": ["concise user-visible diagnosis bullets"],
  "changed_paths": ["repo/relative/path/from/changes.patch"],
  "tests_to_run": ["declared gates or deterministic commands expected to pass"],
  "risk_notes": ["known limitations or assumptions"]
}
```

`changed_paths` must exactly match the repo-relative paths changed by `changes.patch`.  The patch gate rejects mismatches before applying the patch.

## Patch constraints

The patch must be a normal unified git patch rooted at the repo root.  It must touch only the active slice allowed paths and must not touch forbidden paths.  Binary patches are rejected.  Do not mutate reference bundles, `expected.json`, `catalog.json`, `.vc4_auto/**`, or the GPT web driver unless the slice explicitly allows it.

## Executable repo/test/tool contract

Prompt text is not the source of truth; deterministic preflight checks enforce this contract after every patch:

- Changed lit tests must not use unresolved `%tool` tokens.
- A new repo-built tool used in a lit `RUN:` line must have proven lit substitution or proven lit PATH/tool-dir wiring.
- Changed lit config files must be valid Python syntax.
- CMake/tool/test wiring must be local to the slice and allowed by path policy.
- The full declared gates still run after preflight.

Prefer direct build-bin paths in deterministic shell gates, and prefer `%tool` substitutions only when the patch also wires them in lit in a way preflight can detect.


## Deterministic post-patch invariants

After `changes.patch` applies, the autorunner runs cheap preflight invariants before expensive gates:

- `response.json.changed_paths` must exactly match the decoded `changes.patch` paths.
- Changed lit configs such as `lit.cfg.py` and `lit.local.cfg` must parse as Python.
- Every custom `%token` used in a changed lit `RUN:` line must be defined by lit config before `check-vc4` runs.
- Every bare project tool used in a changed lit `RUN:` line must be resolvable through PATH, `compiler/build/bin`, or a CMake tool target introduced by the patch.
- A patch that is already applied is treated as an idempotence/state issue; do not regenerate equivalent add-file patches against an already-landed slice.

These invariants are enforced by deterministic scripts, not by reviewer interpretation. Satisfy them in the patch itself.
