# GPT Pro Output Contract for VC4 Codegen Milestone 1

GPT Pro responses are transported through `pro_scripts/gpt_web_driver.js`. The driver wraps the prompt and requires `BEGIN_GPTWEB_FILE` / `END_GPTWEB_FILE` blocks. It writes the parsed files into a staging directory; the autorun script then validates and applies them.

For normal implementation and fix attempts, GPT Pro must produce exactly these staged files:

```text
response.json
changes.patch
```

No compiler source file should be emitted as a direct GPTWEB file in normal Milestone 1 implementation attempts. Compiler changes must be represented in `changes.patch`.

## Required `response.json` shape

`response.json` must be valid JSON and must use this shape:

```json
{
  "slice_id": "m1-03-minimal-thrend-qasm",
  "attempt_kind": "implementation",
  "summary": ["Short human-readable summary."],
  "diagnosis": ["Why this patch is the right scoped change."],
  "changed_paths": ["repo/relative/path"],
  "tests_added_or_updated": ["repo/relative/path"],
  "expected_gates": ["gate:name"],
  "codex_hint": null,
  "risk_notes": []
}
```

Allowed `attempt_kind` values:

```text
implementation
fix
diagnosis
```

For implementation and fix attempts, `changes.patch` is required. For a diagnosis-only prompt, `changes.patch` is omitted only if the prompt explicitly requests diagnosis-only output.

## Required `changes.patch` shape

`changes.patch` must be a unified git diff relative to the repository root:

```diff
diff --git a/path/to/file b/path/to/file
--- a/path/to/file
+++ b/path/to/file
@@ ...
```

Patch requirements:

- Paths must be repo-relative.
- Paths must not be absolute.
- Paths must not contain `..`.
- Paths must match the current slice's `allowed_paths`.
- Paths must not match the current slice's `forbidden_paths`.
- The patch must apply with `git apply --check`.
- The patch must not include binary blobs.
- The patch must not mutate reference bundles, `expected.json`, or `catalog.json` unless explicitly allowed by the slice.

## GPTWEB block reminder

The final response to the web driver must contain only GPTWEB file blocks. The intended block names are:

```text
BEGIN_GPTWEB_FILE path=response.json
...
END_GPTWEB_FILE

BEGIN_GPTWEB_FILE path=changes.patch
...
END_GPTWEB_FILE
```

The driver may add a token in its wrapper. Obey the exact wrapper format when present.
