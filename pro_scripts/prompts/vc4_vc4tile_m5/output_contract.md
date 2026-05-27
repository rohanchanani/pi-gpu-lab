# VC4 VC4Tile M5 Output Contract

Implementation and failure-fix responses must use downloadable bundle transport `vc4_codegen_download_bundle_v1` when the prompt contains a downloadable bundle contract. Do not use GPTWEB file blocks, pasted patch files, inline full-file dumps, or ad hoc shell heredoc patching.

A valid implementation response starts with compact status JSON whose filenames exactly match the current prompt contract, then exposes exactly the requested downloadable zip and shell script:

```json
{"status":"ok","bundle_zip":"<exact current bundle zip filename>","apply_script":"<exact current shell filename>","source":""}
```

Bundle zip layout:

```text
manifest.json
repo/<repo-relative changed files>
```

`manifest.json` must include `changed_paths`, `diagnosis`, `risk_notes`, and `tests_to_run` as arrays. The bundle must not contain `.vc4_auto/**`, lit output, hardware logs, binary files, symlinks, or unlisted repo files.


## Trusted apply-script rule

The downloadable shell script is not an implementation patch. It must be only the tiny trusted-applier launcher emitted in the current prompt's downloadable artifact contract. It must not contain Python heredocs, unzip/copy logic, chmod/chown logic, backups, file writes, or `rm -rf`. All repo edits must live in `manifest.json` plus `repo/<repo-relative changed files>` inside the zip.
