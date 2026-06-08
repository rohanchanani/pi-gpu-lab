#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'USAGE'
usage: bash compiler/docs/scripts/collect_vc4_post_p13_context.sh [output-dir]

Collect a compact post-P13 source/context package for the next vector-layer
planning chat. The archive contains source, docs, audits, manifests, selected
REPORT.md files, and a final mixed-acceptance result summary when present.
USAGE
  exit 0
fi

if [ ! -d compiler ] || [ ! -d compiler/docs ]; then
  echo "ERROR: run from the pi-gpu-lab repository root" >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_root="${1:-.vc4_auto/post_p13_vector_context_${timestamp}}"
zip_path="${out_root}.zip"

rm -rf "$out_root" "$zip_path"
mkdir -p "$out_root"/{metadata,git,compiler,vc4_auto_reports}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
  fi
}

copy_dir_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
  fi
}

{
  echo "# Post-P13 VC4Kernel Context Manifest"
  echo
  echo "Generated UTC: ${timestamp}"
  echo "Repository: $(pwd)"
  echo
  echo "## Read First"
  echo
  echo "- compiler/docs/vc4kernel_surface_v2_final_lock.md"
  echo "- compiler/docs/codegen/vc4kernel_dialect_strict_specification.md"
  echo "- compiler/docs/vc4kernel_surface_v2_support_matrix.json"
  echo "- compiler/docs/vc4kernel_mixed_acceptance_policy.md"
  echo "- compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json"
  echo "- compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_fixture_claims.json"
  echo "- metadata/final_mixed_acceptance_results.txt"
  echo
  echo "## Boundary"
  echo
  echo "Producer value-layer work enters the locked VC4Kernel surface, then SSAVC4, then scheduled VC4."
  echo "Triton direct-to-VC4Kernel is forbidden; Triton enters through the standard value layer first."
} > "$out_root/metadata/POST_P13_CONTEXT_MANIFEST.md"

git status --short --branch > "$out_root/git/status_short_branch.txt" || true
git log --oneline --decorate -500 > "$out_root/git/log_oneline_decorate_500.txt" || true
git rev-parse HEAD > "$out_root/git/head.txt" || true
git diff --stat > "$out_root/git/diff_stat.txt" || true
git diff --check > "$out_root/git/diff_check.txt" 2>&1 || true

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python3=$(python3 --version 2>&1 || true)"
  echo "cmake=$(cmake --version 2>&1 | head -1 || true)"
  echo "ninja=$(ninja --version 2>&1 || true)"
  echo "llvm_lit=$(command -v llvm-lit || true)"
  echo "vc4_opt=$(compiler/build/bin/vc4-opt --version 2>&1 | head -3 || true)"
  echo "vc4_codegen=$(compiler/build/bin/vc4-codegen --help 2>&1 | head -3 || true)"
} > "$out_root/metadata/tools.txt"

copy_if_exists "README.md" "$out_root/README.md"
copy_if_exists "compiler/CMakeLists.txt" "$out_root/compiler/CMakeLists.txt"
copy_dir_if_exists "compiler/cmake" "$out_root/compiler/cmake"
copy_dir_if_exists "compiler/docs" "$out_root/compiler/docs"

copy_dir_if_exists "compiler/include/vc4/Dialect/VC4Kernel" "$out_root/compiler/include/vc4/Dialect/VC4Kernel"
copy_dir_if_exists "compiler/lib/Dialect/VC4Kernel" "$out_root/compiler/lib/Dialect/VC4Kernel"
copy_dir_if_exists "compiler/include/vc4/Dialect/SSAVC4" "$out_root/compiler/include/vc4/Dialect/SSAVC4"
copy_dir_if_exists "compiler/lib/Dialect/SSAVC4" "$out_root/compiler/lib/Dialect/SSAVC4"
copy_dir_if_exists "compiler/include/vc4/Dialect/VC4" "$out_root/compiler/include/vc4/Dialect/VC4"
copy_dir_if_exists "compiler/lib/Dialect/VC4" "$out_root/compiler/lib/Dialect/VC4"

copy_dir_if_exists "compiler/include/vc4/Conversion/VC4KernelToSSAVC4" "$out_root/compiler/include/vc4/Conversion/VC4KernelToSSAVC4"
copy_dir_if_exists "compiler/lib/Conversion/VC4KernelToSSAVC4" "$out_root/compiler/lib/Conversion/VC4KernelToSSAVC4"
copy_dir_if_exists "compiler/include/vc4/Conversion/SSAVC4ToVC4" "$out_root/compiler/include/vc4/Conversion/SSAVC4ToVC4"
copy_dir_if_exists "compiler/lib/Conversion/SSAVC4ToVC4" "$out_root/compiler/lib/Conversion/SSAVC4ToVC4"

copy_dir_if_exists "compiler/include/vc4/Target/VC4" "$out_root/compiler/include/vc4/Target/VC4"
copy_dir_if_exists "compiler/lib/Target/VC4" "$out_root/compiler/lib/Target/VC4"
copy_dir_if_exists "compiler/include/vc4/Support" "$out_root/compiler/include/vc4/Support"
copy_dir_if_exists "compiler/lib/Support" "$out_root/compiler/lib/Support"

copy_dir_if_exists "compiler/test/Dialect/VC4Kernel" "$out_root/compiler/test/Dialect/VC4Kernel"
copy_dir_if_exists "compiler/test/Dialect/SSAVC4" "$out_root/compiler/test/Dialect/SSAVC4"
copy_dir_if_exists "compiler/test/Dialect/VC4" "$out_root/compiler/test/Dialect/VC4"
copy_dir_if_exists "compiler/test/Conversion/VC4KernelToSSAVC4" "$out_root/compiler/test/Conversion/VC4KernelToSSAVC4"
copy_dir_if_exists "compiler/test/Conversion/SSAVC4ToVC4" "$out_root/compiler/test/Conversion/SSAVC4ToVC4"
copy_dir_if_exists "compiler/test/CodeGen/VC4Kernel/Support" "$out_root/compiler/test/CodeGen/VC4Kernel/Support"
copy_dir_if_exists "compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance" "$out_root/compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance"
copy_dir_if_exists "compiler/test/CodeGen/VC4Kernel/Hardware/Run" "$out_root/compiler/test/CodeGen/VC4Kernel/Hardware/Run"
copy_dir_if_exists "compiler/test/CodeGen/SSAVC4/Hardware/Run" "$out_root/compiler/test/CodeGen/SSAVC4/Hardware/Run"
copy_dir_if_exists "compiler/test/CodeGen/SSAVC4" "$out_root/compiler/test/CodeGen/SSAVC4"
copy_dir_if_exists "compiler/test/CodeGen/VC4" "$out_root/compiler/test/CodeGen/VC4"
copy_if_exists "compiler/test/lit.cfg.py" "$out_root/compiler/test/lit.cfg.py"
copy_if_exists "compiler/test/lit.site.cfg.py.in" "$out_root/compiler/test/lit.site.cfg.py.in"

state_root=".vc4_auto/codegen_p13h_final_mixed_acceptance"
{
  echo "# P13h Final Mixed Acceptance Results"
  echo
  if [ -d "$state_root/hardware" ]; then
    find "$state_root/hardware" -path '*/candidate_work/.vc4_candidate_run_attempt.log' -type f -print0 \
      | sort -z \
      | xargs -0 grep -h '^VC4_TEST_RESULT ' || true
  else
    echo "No P13h state root found at ${state_root}."
  fi
} > "$out_root/metadata/final_mixed_acceptance_results.txt"

{
  echo "# P13h Result Summary"
  total="$(grep -c '^VC4_TEST_RESULT ' "$out_root/metadata/final_mixed_acceptance_results.txt" || true)"
  pass="$(grep '^VC4_TEST_RESULT ' "$out_root/metadata/final_mixed_acceptance_results.txt" | grep -c ' status=PASS ' || true)"
  echo "total_result_lines=${total}"
  echo "pass_result_lines=${pass}"
  grep '^VC4_TEST_RESULT ' "$out_root/metadata/final_mixed_acceptance_results.txt" \
    | grep -Ev ' status=PASS |total_mismatches=0|sentinel_mismatches=0|launch_failures=0' || true
} > "$out_root/metadata/final_mixed_acceptance_summary.txt"

if [ -d .vc4_auto ]; then
  find .vc4_auto -maxdepth 3 -type f -name REPORT.md \
    \( -path '.vc4_auto/p9_*' -o -path '.vc4_auto/p10_*' -o -path '.vc4_auto/p11_*' -o -path '.vc4_auto/p12_*' -o -path '.vc4_auto/p13_*' \) \
    | sort > "$out_root/metadata/selected_report_files.txt" || true
  while IFS= read -r report; do
    [ -f "$report" ] || continue
    rel="${report#.vc4_auto/}"
    mkdir -p "$out_root/vc4_auto_reports/$(dirname "$rel")"
    cp "$report" "$out_root/vc4_auto_reports/$rel"
  done < "$out_root/metadata/selected_report_files.txt"
fi

{
  echo "# Source Surface Scan"
  rg -n 'vc4kernel\.|ssavc4\.|vc4\.|fragment_|vpm_|vdr_|vdw_|safe_offset|inactive_|subword_selector|word|selector|resource' \
    compiler/include/vc4 compiler/lib compiler/test/Dialect compiler/test/Conversion compiler/test/CodeGen/VC4Kernel compiler/docs 2>/dev/null || true
} > "$out_root/metadata/source_surface_scan.txt"

find "$out_root" -type d \( -name build -o -name candidate_work -o -name kernels -o -name objs -o -name lowered -o -name hardware -o -name __pycache__ -o -name node_modules \) -prune -exec rm -rf {} + 2>/dev/null || true
find "$out_root" -type f \( -iname '*.ttf' -o -iname '*.otf' -o -iname '*.woff' -o -iname '*.woff2' \) -delete 2>/dev/null || true
find "$out_root" -type f | sort > "$out_root/metadata/file_list.txt"
du -sh "$out_root" > "$out_root/metadata/size.txt" || true

python3 - <<'PY' "$out_root" "$zip_path"
import hashlib
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
zip_path = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(root.parent))
digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
(root / "metadata" / "zip_sha256.txt").write_text(digest + "\n")
print(f"POST_P13_CONTEXT_DIR={root}")
print(f"POST_P13_CONTEXT_ZIP={zip_path}")
print(f"POST_P13_CONTEXT_SHA256={digest}")
PY
