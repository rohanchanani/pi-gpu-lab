# VC4 Codegen M2 Output Contract

Implementation and failure-fix responses must use the downloadable bundle transport `vc4_codegen_download_bundle_v1`. Do not use GPTWEB file blocks, inline patches, or pasted full files for implementation attempts when a downloadable bundle contract is present.

The current prompt's `DOWNLOAD_CONTRACT_JSON` is the only source of truth for artifact filenames. Never copy `artifact_prefix`, `bundle_zip`, `apply_script`, timestamps, or attempt numbers from prior assistant messages, failure packets, `artifact_transport.json`, or chat history. If the current prompt says `Attempt: 3`, the generated filenames and bundle manifest must use attempt 3, not attempt 1.

The final visible response must start with exactly this shape, using the exact current filenames from `DOWNLOAD_CONTRACT_JSON`:

```json
{"status":"ok","bundle_zip":"<exact current bundle zip filename>","apply_script":"<exact current shell filename>","source":""}
```

Then expose exactly two downloadable files with visible labels exactly equal to those filenames.

## Bundle zip layout

```text
manifest.json
repo/<repo-relative changed files>
```

`manifest.json` must be a UTF-8 JSON object with this shape:

```json
{
  "schema_version": 1,
  "transport": "vc4_codegen_download_bundle_v1",
  "slice_id": "<active slice id>",
  "attempt": 1,
  "diagnosis": ["concise reason for this change"],
  "risk_notes": [],
  "tests_to_run": [],
  "changed_paths": [
    {
      "path": "compiler/lib/Target/VC4/VC4ArtifactEmitter.cpp",
      "action": "write",
      "mode": "0644",
      "sha256": "<sha256 of repo/compiler/lib/Target/VC4/VC4ArtifactEmitter.cpp bytes>"
    }
  ]
}
```

Rules:

- `attempt` must equal the current prompt's attempt number.
- `diagnosis`, `risk_notes`, and `tests_to_run` must be arrays, not strings.
- `changed_paths[].path` must be repo-relative without a leading `repo/` prefix.
- Each `write` entry must have a matching zip member at `repo/<path>`.
- The zip must not contain `.vc4_auto/**`, lit `Output/**`, `.lit_test_times.txt`, hardware `run.log`, binary files, symlinks, or extra repo members not listed in `changed_paths`.
- Do not mutate reference bundles, `expected.json`, catalog files, or GPT/web-driver files.

## Apply shell script

The shell script must be a tiny trusted-applier launcher only. It must delegate to:

```text
pro_scripts/vc4_codegen_download_bundle_apply.py validate-apply
```

It must include `--slice <active slice id>`, `--bundle <current exact zip filename>`, `--apply-script <current exact shell filename>`, and `--expect-attempt <current attempt>`. Do not put Python patching logic, heredocs, file writes, unzip/copy logic, or `rm -rf` in the shell script.

The central typed verifier remains the source of truth. A slice is incomplete until its M2 typed verification entry passes.
