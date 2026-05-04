# VC4 Codegen Milestone 1 Automation

This directory contains the deterministic workflow contracts for Milestone 1 of VC4 code generation.

## Milestone 1 goal

Generate a candidate hardware bundle from a final-stage scheduled VC4 kernel:

```bash
vc4-codegen input.mlir --emit-bundle candidate/
```

The generated candidate bundle eventually contains:

```text
candidate/kernel.qasm
candidate/kernel_launch.c
candidate/kernel_launch.h
candidate/manifest.json
```

Milestone 1 is complete when generated candidates build and run on hardware for `minimal_thrend`-like and simple memory-output-like tests.

## Repo layout assumptions

The automation scripts live under:

```text
pro_scripts/
```

The existing GPT web driver is:

```text
pro_scripts/gpt_web_driver.js
```

The compiler build directory is:

```text
compiler/build
```

The expected existing build commands are:

```bash
ninja -C compiler/build vc4-opt
ninja -C compiler/build check-vc4
```

The first hardware smoke fixture is expected at:

```text
compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/
```

## Stage 1 contents

Stage 1 installs static workflow files only:

```text
pro_scripts/vc4_codegen_m1_worklist.json
pro_scripts/vc4_codegen_m1_context_profiles.json
pro_scripts/prompts/vc4_codegen_m1/constitution.md
pro_scripts/prompts/vc4_codegen_m1/output_contract.md
pro_scripts/prompts/vc4_codegen_m1/slice_contract.md
pro_scripts/prompts/vc4_codegen_m1/codex_contract.md
pro_scripts/README_vc4_codegen_m1.md
```

Stage 1 does not run GPT Pro, Codex, compiler builds, or hardware.

## Workflow authority

The future autorun script owns deterministic control:

- slice selection
- dependency checking
- context pack generation
- prompt rendering
- GPT web-driver invocation
- patch validation
- gate execution
- failure classification
- Codex dispatch
- state recording
- auto-commit after passing slices

GPT Pro writes substantive patches for one declared slice at a time. Codex handles only classified mechanical failures.

## Important invariants

Do not mutate reference bundles during Milestone 1 codegen work:

```text
compiler/test/CodeGen/VC4/Hardware/Run/*/reference/**
compiler/test/CodeGen/VC4/Hardware/Run/*/expected.json
compiler/test/CodeGen/VC4/catalog.json
```

Do not use exact qasm text matching as the success criterion. Candidate bundles must pass the same semantic oracle as the reference side.

The uploaded `expected.json` schema used by existing tests has top-level `name` and `status`, optional exact-match fields under `required`, and optional tolerance groups such as `float_max`.

## Stage 1 verification

From the repo root, run:

```bash
python3 -m json.tool pro_scripts/vc4_codegen_m1_worklist.json >/tmp/vc4_m1_worklist.pretty.json
python3 -m json.tool pro_scripts/vc4_codegen_m1_context_profiles.json >/tmp/vc4_m1_context_profiles.pretty.json

python3 - <<'PY'
import json
from pathlib import Path
worklist = json.loads(Path('pro_scripts/vc4_codegen_m1_worklist.json').read_text())
profiles = json.loads(Path('pro_scripts/vc4_codegen_m1_context_profiles.json').read_text())
slices = worklist['slices']
ids = [s['id'] for s in slices]
if len(ids) != len(set(ids)):
    raise SystemExit('duplicate slice ids')
missing_deps = []
for s in slices:
    for dep in s.get('depends_on', []):
        if dep not in ids:
            missing_deps.append((s['id'], dep))
if missing_deps:
    raise SystemExit(f'missing deps: {missing_deps}')
profile_names = set(profiles['profiles'])
missing_profiles = [s['context_profile'] for s in slices if s['context_profile'] not in profile_names]
if missing_profiles:
    raise SystemExit(f'missing profiles: {missing_profiles}')
print(f'OK: {len(slices)} slices')
print('Slice order:')
for i, s in enumerate(slices, 1):
    print(f'  {i:02d}. {s["id"]}: {s["title"]}')
print(f'OK: {len(profile_names)} context profiles')
PY

find pro_scripts -maxdepth 4 \
  \( -name 'vc4_codegen_m1_*' -o -name 'README_vc4_codegen_m1.md' -o -path '*/prompts/vc4_codegen_m1/*' \) \
  -print | sort

git diff --stat -- pro_scripts
```

Commit after the verification succeeds:

```bash
git add pro_scripts/vc4_codegen_m1_worklist.json \
        pro_scripts/vc4_codegen_m1_context_profiles.json \
        pro_scripts/prompts/vc4_codegen_m1 \
        pro_scripts/README_vc4_codegen_m1.md

git commit -m "add vc4 codegen milestone 1 workflow contracts"
```
