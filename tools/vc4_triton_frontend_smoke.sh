#!/usr/bin/env bash
set -euo pipefail

allow_missing_examples=0
if [[ "${1:-}" == "--allow-missing-examples" ]]; then
  allow_missing_examples=1
elif [[ $# -gt 0 ]]; then
  echo "usage: tools/vc4_triton_frontend_smoke.sh [--allow-missing-examples]" >&2
  exit 2
fi

if [[ -x .vc4_auto/triton_phase6_venv/bin/python ]]; then
  PY=.vc4_auto/triton_phase6_venv/bin/python
else
  PY=python3
fi

"$PY" - <<'PY'
try:
    import triton
except Exception as exc:
    raise SystemExit(
        "Triton is not installed for optional Phase 6 frontend smoke. "
        "Use tools/requirements-triton-phase6.txt or the pinned v3.7.0 source fallback. "
        f"Import failed: {exc}"
    )
version = getattr(triton, "__version__", "<missing>")
if not str(version).startswith("3.7."):
    raise SystemExit(f"Expected Triton 3.7.x for Phase 6, found {version}")
print("TRITON_VERSION=" + str(version))
PY

example=compiler/examples/triton/vector_add.py
if [[ ! -f "$example" ]]; then
  if [[ "$allow_missing_examples" -eq 1 ]]; then
    echo "Phase 6c Triton examples are not present; skipping frontend smoke by request."
    exit 0
  fi
  echo "Missing Phase 6c example: $example" >&2
  echo "Rerun with --allow-missing-examples before Phase 6c." >&2
  exit 1
fi

out_dir=.vc4_auto/triton_frontend_smoke
mkdir -p "$out_dir"
"$PY" tools/vc4_emit_ttir.py "$example" \
  --kernel-name vector_add_kernel \
  --signature '*fp32,*fp32,*fp32,i32,16' \
  --target cuda:80:32 \
  --num-warps 1 \
  --num-stages 3 \
  --out "$out_dir/vector_add.ttir.mlir" \
  --metadata-out "$out_dir/vector_add.metadata.json"
"$PY" tools/vc4_parse_ttir.py "$out_dir/vector_add.ttir.mlir" \
  --target cuda:80:32 \
  --summary-json "$out_dir/vector_add.summary.json"

