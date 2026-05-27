#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-$PWD}"
ZIP_PATH="${2:-$HOME/Downloads/m5_automation_repair_patch.zip}"
if [ ! -f "$REPO_ROOT/compiler/CMakeLists.txt" ]; then
  echo "repo root must contain compiler/CMakeLists.txt: $REPO_ROOT" >&2
  exit 1
fi
if [ ! -f "$ZIP_PATH" ]; then
  echo "missing patch zip: $ZIP_PATH" >&2
  exit 1
fi
python3 - "$ZIP_PATH" "$REPO_ROOT" <<'PYAPPLY'
from pathlib import Path
import hashlib, json, sys, zipfile
zip_path = Path(sys.argv[1]).expanduser().resolve()
repo = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    for name in names:
        if name.startswith('/') or '..' in Path(name).parts:
            raise SystemExit(f'unsafe zip member: {name}')
    manifest = json.loads(zf.read('manifest.json'))
    changed = manifest.get('changed_paths')
    if not isinstance(changed, list) or not changed:
        raise SystemExit('manifest changed_paths must be a non-empty list')
    expected = {entry['path']: entry['sha256'] for entry in manifest.get('files', [])}
    for rel in changed:
        if rel.startswith('/') or '..' in Path(rel).parts or rel.startswith('.vc4_auto/'):
            raise SystemExit(f'unsafe changed path: {rel}')
        member = 'repo/' + rel
        if member not in names:
            raise SystemExit(f'missing payload member: {member}')
        data = zf.read(member)
        digest = hashlib.sha256(data).hexdigest()
        if expected.get(rel) != digest:
            raise SystemExit(f'sha256 mismatch for {rel}')
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
print('applied M5 automation repair patch')
PYAPPLY
