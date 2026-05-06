# VC4 Codegen Milestone 1 GPT Output Contract

Implementation and failure-fix prompts use the **downloadable bundle transport**. Do not paste compiler source, C++ string literals, shell scripts, or unified diffs into chat text.

The downloadable transport is named:

```text
vc4_codegen_download_bundle_v1
```

The active prompt gives exact filenames for two downloadable artifacts:

1. a zip file
2. a tiny shell launcher

The local driver downloads or collects those exact files, then validates and applies the zip with trusted local code. The GPT-generated shell launcher is validated as a narrow manual-recovery launcher; it is not trusted for arbitrary repo mutation.

## Zip layout

The zip must contain exactly this source-bearing layout:

```text
manifest.json
repo/<repo-relative changed file 1>
repo/<repo-relative changed file 2>
...
```

No absolute paths, no parent-directory components, no symlinks, no `.git/**`, no `.vc4_auto/**`, and no hidden source-bearing files outside `repo/`.

## `manifest.json` schema

```json
{
  "schema_version": 1,
  "transport": "vc4_codegen_download_bundle_v1",
  "slice_id": "active slice id",
  "slice_title": "active slice title",
  "attempt": 1,
  "changed_paths": [
    {
      "path": "repo/relative/path",
      "action": "write",
      "sha256": "64 lowercase hex characters of repo/<path> bytes",
      "mode": "0644"
    }
  ],
  "deleted_paths": [],
  "summary": "one-sentence patch summary",
  "diagnosis": ["concise user-visible diagnosis bullets"],
  "tests_to_run": ["declared gates or deterministic commands expected to pass"],
  "risk_notes": ["known limitations or assumptions"]
}
```

`changed_paths` may include `action = "write"` or `action = "delete"`. Write entries must have a corresponding file at `repo/<path>` in the zip. Delete entries must not have a corresponding file. `deleted_paths` is optional and is treated as extra delete entries.

The local applier rejects the bundle unless:

- `transport` is exactly `vc4_codegen_download_bundle_v1`.
- `slice_id` matches the active slice.
- every path is repo-relative and allowlisted for the active slice.
- no path matches the forbidden-path policy.
- every listed write file exists in the zip and its SHA-256 matches the manifest.
- no unlisted `repo/` file exists in the zip.
- file modes are only `0644` or `0755`.
- file contents are text-like and do not contain NUL bytes.

## Shell launcher

The shell launcher must be tiny and must delegate to the trusted local applier. It must not contain source code. It must not run `rm -rf`, network commands, `eval`, `source`, Python one-liners, or arbitrary repo mutation.

Expected shape:

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-$(pwd)}"
BUNDLE_ZIP="${HOME}/Downloads/<exact bundle zip filename>"
APPLY_SCRIPT="${HOME}/Downloads/<exact shell filename>"
python3 "${REPO_ROOT}/pro_scripts/vc4_codegen_download_bundle_apply.py" validate-apply \
  --repo "${REPO_ROOT}" \
  --slice "<active slice id>" \
  --bundle "${BUNDLE_ZIP}" \
  --apply-script "${APPLY_SCRIPT}" \
  --expect-attempt "<attempt>"
```


## Download link labels

The final visible ChatGPT response must expose both downloadable artifacts with link/attachment labels that are byte-for-byte equal to the exact filenames from the active prompt.

Required final response shape:

```json
{"status":"ok","bundle_zip":"<exact bundle zip filename>","apply_script":"<exact shell filename>","source":""}
```

Then show exactly these two downloadable links/attachments, with no generic labels:

```text
<exact bundle zip filename>
<exact shell filename>
```

Do not label the links `Download bundle zip`, `Download apply script`, `bundle`, `script`, or any other descriptive text. The local browser driver matches the unique filenames from the JSON contract to avoid clicking stale links in accumulated chats.

## Legacy fallback

Older tooling may still accept `response.json` + `changes.patch` GPTWEB file blocks, but new implementation and failure-fix prompts should not use that path. The bundle transport exists specifically to avoid quote, backslash, markdown, and patch-rendering corruption.
