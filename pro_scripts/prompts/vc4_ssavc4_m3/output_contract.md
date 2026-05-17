# VC4 SSAVC4 M3 Output Contract

Implementation and failure-fix responses must use the downloadable bundle transport `vc4_codegen_download_bundle_v1` when the prompt contains a downloadable bundle contract. Do not use GPTWEB file blocks, pasted patch files, or inline full-file dumps for implementation attempts.

The current prompt's `DOWNLOAD_CONTRACT_JSON` is the only source of truth for artifact filenames, attempt number, slice id, and timestamp. Never copy `artifact_prefix`, `bundle_zip`, or `apply_script` values from prior assistant messages, failure packets, old bundle manifests, or chat history.

A valid implementation response starts with a compact status JSON whose filenames exactly match the current prompt contract, then exposes exactly the requested downloadable zip and shell script:

```json
{"status":"ok","bundle_zip":"<exact current bundle zip filename>","apply_script":"<exact current shell filename>","source":""}
```

## Bundle zip layout

```text
manifest.json
repo/<repo-relative changed files>
```

`manifest.json` must be UTF-8 JSON with:

```json
{
  "schema_version": 1,
  "transport": "vc4_codegen_download_bundle_v1",
  "slice_id": "<active slice id>",
  "attempt": 1,
  "diagnosis": ["concise reason for the change"],
  "risk_notes": [],
  "tests_to_run": [],
  "changed_paths": [
    {
      "path": "compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Ops.td",
      "action": "write",
      "mode": "0644",
      "sha256": "<sha256 of repo/<path> bytes>"
    }
  ]
}
```

Rules:

- `diagnosis`, `risk_notes`, and `tests_to_run` are arrays.
- `changed_paths[].path` is repo-relative and has no leading `repo/`.
- Every `write` entry has a matching zip member at `repo/<path>`.
- The zip must not contain `.vc4_auto/**`, lit output, `.lit_test_times.txt`, hardware `run.log`, binary files, symlinks, or extra repo members not listed in `changed_paths`.
- The shell script must be a tiny launcher for `pro_scripts/vc4_codegen_download_bundle_apply.py validate-apply` and must pass `--repo "$REPO_DIR"`, `--slice-id <current slice id>`, `--attempt <current attempt number>`, `--expect-attempt <current attempt number>`, `--bundle <exact current bundle filename>`, and `--apply-script <exact current shell filename>`.

For M3 specifically, implementation bundles must not include M2 reference-bundle mutations, `generated_examples` mutations, generic driver rewrites, or direct `gpu -> vc4` lowering.
