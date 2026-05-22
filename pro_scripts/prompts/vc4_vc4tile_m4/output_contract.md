# VC4 VC4Tile M4 Output Contract

Implementation and failure-fix responses must use the downloadable bundle transport `vc4_codegen_download_bundle_v1` when the prompt contains a downloadable bundle contract. Do not use GPTWEB file blocks, pasted patch files, or inline full-file dumps for implementation attempts.

A valid implementation response starts with compact status JSON whose filenames exactly match the current prompt contract, then exposes exactly the requested downloadable zip and shell script:

```json
{"status":"ok","bundle_zip":"<exact current bundle zip filename>","apply_script":"<exact current shell filename>","source":""}
```

## Bundle zip layout

```text
manifest.json
repo/<repo-relative changed files>
```

Rules:

- All changed paths are repo-relative and listed in `manifest.json`.
- The zip must not contain `.vc4_auto/**`, lit output, `.lit_test_times.txt`, hardware `run.log`, binary files, symlinks, or extra repo members not listed in `changed_paths`.
- The shell script must be a tiny launcher for `pro_scripts/vc4_codegen_download_bundle_apply.py validate-apply` and must pass the current slice id, attempt number, bundle filename, and apply script filename.
- M4 bundles must not include producer lowering from Triton, IREE, StableHLO, Torch, JAX, PyTorch, or upstream MLIR `gpu`.
- M4 bundles must not include direct `vc4tile -> vc4` lowering, direct producer-to-SSAVC4 shortcuts, expected JSON rewrites, reference bundle rewrites, generic driver rewrites, or generated `.vc4_auto` outputs.
