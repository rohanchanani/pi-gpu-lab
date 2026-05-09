# VC4 Codegen M2 Output Contract

Patch-producing responses must stage a downloadable bundle using the existing `vc4_codegen_download_bundle_v1` transport unless the prompt explicitly asks for prose only.

The bundle layout is:

```text
manifest.json
repo/<repo-relative changed files>
```

Patches must be repo-relative, text-only, and path-policy compliant. Do not include generated `.vc4_auto/**`, lit `Output/**`, `.lit_test_times.txt`, hardware `run.log`, or binary files.

Every implementation response must include:

1. a concise diagnosis,
2. the contract being satisfied,
3. exact changed files,
4. commands to run,
5. expected verifier IDs that should pass.

The central typed verifier remains the source of truth. A slice is incomplete until its M2 typed verification entry passes.
