#!/usr/bin/env bash
set -euo pipefail

# VC4 / VC4Kernel -> Vector/Triton PRE-P9 context collector.
#
# Run this from the repo root of rohanchanani/pi-gpu-lab after P8.5 succeeds
# and before P9 starts. This is intended for preliminary vector/Triton design
# discussions while P9-P13 continue in the original chat.
#
# It may also be rerun with --post-p13 after final surface lock; the manifest
# will then mark the timing differently, but the included content is the same
# style of repo snapshot.
#
# Output:
#   .vc4_auto/vector_triton_pre_p9_context_<timestamp>.zip

MODE="pre-p9"
if [ "${1:-}" = "--post-p13" ]; then
  MODE="post-p13"
elif [ "${1:-}" != "" ]; then
  echo "usage: $0 [--post-p13]" >&2
  exit 2
fi

if [ ! -d compiler ]; then
  echo "ERROR: run from pi-gpu-lab repo root (expected ./compiler)" >&2
  exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="$(pwd)"
OUT_ROOT=".vc4_auto/vector_triton_${MODE}_context_${TS}"
ZIP_PATH="${OUT_ROOT}.zip"

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"/{metadata,git,compiler,libpi,grep,vc4_auto_reports}

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

# Metadata manifest
{
  echo "# VC4 Vector/Triton ${MODE} Context Manifest"
  echo
  echo "Generated UTC: ${TS}"
  echo "Repo root: ${ROOT}"
  echo "Mode: ${MODE}"
  echo
  if [ "$MODE" = "pre-p9" ]; then
    echo "This context is intended after P8.5 and before P9."
    echo "Use it for preliminary vector/Triton design discussions, not final implementation packages."
    echo "P9-P13 are expected to still be in progress or pending."
  else
    echo "This context is intended after P13 final VC4Kernel surface lock."
    echo "It may be used for actual vector/Triton implementation package planning."
  fi
  echo
  echo "## First files to read"
  echo
  echo "- compiler/docs/vc4kernel_dialect_strict_specification.md"
  echo "- compiler/docs/vc4kernel_surface_v2_support_matrix.json"
  echo "- compiler/docs/vc4kernel_mixed_acceptance_policy.md if present"
  echo "- compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelOps.td"
  echo "- compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.td"
  echo "- compiler/lib/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.cpp"
  echo "- compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp"
  echo "- compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance/mixed_acceptance_manifest.json if present"
} > "$OUT_ROOT/metadata/PRE_P9_CONTEXT_MANIFEST.md"

# Git state
git status --short --branch > "$OUT_ROOT/git/status_short_branch.txt" || true
git status > "$OUT_ROOT/git/status.txt" || true
git log --oneline --decorate -300 > "$OUT_ROOT/git/log_oneline_decorate_300.txt" || true
git rev-parse HEAD > "$OUT_ROOT/git/head.txt" || true
git branch --show-current > "$OUT_ROOT/git/branch.txt" || true
git diff --stat > "$OUT_ROOT/git/diff_stat.txt" || true
git diff > "$OUT_ROOT/git/diff.patch" || true
git diff --check > "$OUT_ROOT/git/diff_check.txt" 2>&1 || true
git ls-files > "$OUT_ROOT/git/ls_files.txt" || true

# Tool metadata
{
  echo "# Tool metadata"
  echo
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "mode=${MODE}"
  echo "pwd=$(pwd)"
  echo "uname=$(uname -a || true)"
  echo "python3=$(python3 --version 2>&1 || true)"
  echo "cmake=$(cmake --version 2>&1 | head -1 || true)"
  echo "ninja=$(ninja --version 2>&1 || true)"
  echo "llvm-lit=$(command -v llvm-lit || true)"
  echo "vc4-opt=$(./compiler/build/bin/vc4-opt --version 2>&1 | head -3 || true)"
  echo "vc4-codegen=$(./compiler/build/bin/vc4-codegen --help 2>&1 | head -3 || true)"
} > "$OUT_ROOT/metadata/tools.txt"

# Docs and top-level metadata
copy_if_exists "README.md" "$OUT_ROOT/README.md"
copy_if_exists "compiler/README.md" "$OUT_ROOT/compiler/README.md"
copy_dir_if_exists "compiler/docs" "$OUT_ROOT/compiler/docs"

# Dialect definitions and implementations
copy_dir_if_exists "compiler/include/vc4/Dialect/VC4Kernel" "$OUT_ROOT/compiler/include/vc4/Dialect/VC4Kernel"
copy_dir_if_exists "compiler/lib/Dialect/VC4Kernel" "$OUT_ROOT/compiler/lib/Dialect/VC4Kernel"
copy_dir_if_exists "compiler/include/vc4/Dialect/SSAVC4" "$OUT_ROOT/compiler/include/vc4/Dialect/SSAVC4"
copy_dir_if_exists "compiler/lib/Dialect/SSAVC4" "$OUT_ROOT/compiler/lib/Dialect/SSAVC4"
copy_dir_if_exists "compiler/include/vc4/Dialect/VC4" "$OUT_ROOT/compiler/include/vc4/Dialect/VC4"
copy_dir_if_exists "compiler/lib/Dialect/VC4" "$OUT_ROOT/compiler/lib/Dialect/VC4"

# Conversion passes
copy_dir_if_exists "compiler/lib/Conversion/VC4KernelToSSAVC4" "$OUT_ROOT/compiler/lib/Conversion/VC4KernelToSSAVC4"
copy_dir_if_exists "compiler/include/vc4/Conversion/VC4KernelToSSAVC4" "$OUT_ROOT/compiler/include/vc4/Conversion/VC4KernelToSSAVC4"
copy_dir_if_exists "compiler/lib/Conversion/SSAVC4ToVC4" "$OUT_ROOT/compiler/lib/Conversion/SSAVC4ToVC4"
copy_dir_if_exists "compiler/include/vc4/Conversion/SSAVC4ToVC4" "$OUT_ROOT/compiler/include/vc4/Conversion/SSAVC4ToVC4"

# Future/upper-layer conversions if present
{
  echo "# Possible upper-layer conversion dirs/files"
  find compiler/lib compiler/include compiler/test -maxdepth 6 \( -type d -o -type f \) 2>/dev/null \
    | grep -Ei 'Vector|Triton|TTIR|TTGIR|Linalg|MemRef|Arith|Math|ToVC4Kernel|VC4KernelTo' || true
} > "$OUT_ROOT/metadata/possible_upper_layer_paths.txt"

# Targets/runtime/artifact emitter/support
copy_dir_if_exists "compiler/lib/Target/VC4" "$OUT_ROOT/compiler/lib/Target/VC4"
copy_dir_if_exists "compiler/include/vc4/Target/VC4" "$OUT_ROOT/compiler/include/vc4/Target/VC4"
copy_dir_if_exists "compiler/include/vc4/Support" "$OUT_ROOT/compiler/include/vc4/Support"
copy_dir_if_exists "compiler/lib/Support" "$OUT_ROOT/compiler/lib/Support"

# Tests and support scripts
copy_dir_if_exists "compiler/test/Dialect/VC4Kernel" "$OUT_ROOT/compiler/test/Dialect/VC4Kernel"
copy_dir_if_exists "compiler/test/Dialect/SSAVC4" "$OUT_ROOT/compiler/test/Dialect/SSAVC4"
copy_dir_if_exists "compiler/test/Dialect/VC4" "$OUT_ROOT/compiler/test/Dialect/VC4"
copy_dir_if_exists "compiler/test/Conversion/VC4KernelToSSAVC4" "$OUT_ROOT/compiler/test/Conversion/VC4KernelToSSAVC4"
copy_dir_if_exists "compiler/test/Conversion/SSAVC4ToVC4" "$OUT_ROOT/compiler/test/Conversion/SSAVC4ToVC4"
copy_dir_if_exists "compiler/test/CodeGen/VC4Kernel" "$OUT_ROOT/compiler/test/CodeGen/VC4Kernel"
copy_dir_if_exists "compiler/test/CodeGen/SSAVC4" "$OUT_ROOT/compiler/test/CodeGen/SSAVC4"
copy_dir_if_exists "compiler/test/CodeGen/VC4" "$OUT_ROOT/compiler/test/CodeGen/VC4"
copy_if_exists "compiler/test/lit.cfg.py" "$OUT_ROOT/compiler/test/lit.cfg.py"
copy_if_exists "compiler/test/lit.site.cfg.py.in" "$OUT_ROOT/compiler/test/lit.site.cfg.py.in"

# Build/config scripts, without build outputs
copy_if_exists "compiler/CMakeLists.txt" "$OUT_ROOT/compiler/CMakeLists.txt"
copy_dir_if_exists "compiler/cmake" "$OUT_ROOT/compiler/cmake"
copy_dir_if_exists "compiler/utils" "$OUT_ROOT/compiler/utils"
copy_dir_if_exists "scripts" "$OUT_ROOT/scripts"

# Reports/manifests from .vc4_auto but avoid huge hardware artifacts.
if [ -d .vc4_auto ]; then
  find .vc4_auto -maxdepth 5 -type f \( \
      -name 'REPORT.md' -o \
      -name '*report*.md' -o \
      -name '*manifest*.json' -o \
      -name '*acceptance*.json' -o \
      -name '*surface*.json' -o \
      -name '*mixed*.json' -o \
      -name '*.txt' \
    \) -not -path '*/candidate_work/*' -not -path '*/hardware/*' -not -path '*/kernels/*' \
    | sort > "$OUT_ROOT/metadata/vc4_auto_report_files.txt" || true

  while IFS= read -r f; do
    [ -f "$f" ] || continue
    rel="${f#.vc4_auto/}"
    mkdir -p "$OUT_ROOT/vc4_auto_reports/$(dirname "$rel")"
    cp "$f" "$OUT_ROOT/vc4_auto_reports/$rel" || true
  done < "$OUT_ROOT/metadata/vc4_auto_report_files.txt"
fi

# Grep summaries
{
  echo "# VC4Kernel op names and attrs"
  rg -n 'vc4kernel\.[A-Za-z0-9_\.]+|memory_path|coherency|inactive_load|inactive_store|safe_offset|fragment_alu|fragment_reduce|fragment_cmp|fragment_const|fragment_bitcast|fragment_pack|fragment_unpack|sfu|rotate' \
    compiler/include/vc4/Dialect/VC4Kernel compiler/lib/Dialect/VC4Kernel compiler/lib/Conversion/VC4KernelToSSAVC4 compiler/test/Dialect/VC4Kernel compiler/test/Conversion/VC4KernelToSSAVC4 compiler/test/CodeGen/VC4Kernel 2>/dev/null || true
} > "$OUT_ROOT/grep/vc4kernel_surface_scan.txt"

{
  echo "# Boundary/forbidden scan"
  rg -n 'VC4KernelToVC4|vc4tile|tile_broadcast|tile_dot|tile_matmul|tile_contract|fragment_contract|fragment_add|fragment_sub|fragment_mul|fragment_shl|old TMU|safe-address inference|ldtmu0.*spill|spill.*tmu' \
    compiler/include compiler/lib compiler/test compiler/docs 2>/dev/null || true
} > "$OUT_ROOT/grep/boundary_forbidden_scan.txt"

{
  echo "# Upper layer search"
  rg -n 'vector\.|memref\.|scf\.|tt\.|ttg\.|triton|Triton|VectorToVC4Kernel|convert-vector|vector-to-vc4kernel|vc4kernel' \
    compiler/include compiler/lib compiler/test compiler/docs 2>/dev/null || true
} > "$OUT_ROOT/grep/upper_layer_search.txt"

{
  echo "# Status/hardware summaries in reports"
  rg -n 'VC4_TEST_RESULT|READY_FOR_|ACCEPTED=YES|status=PASS|status=FAIL|CHECKPOINT|MIXED_ACCEPTANCE|P8|P9|P10|P11|P12|P13' \
    .vc4_auto compiler/docs compiler/test 2>/dev/null | head -3000 || true
} > "$OUT_ROOT/grep/status_and_hardware_summaries.txt"

{
  echo "# Mixed acceptance files"
  find compiler/test/CodeGen/VC4Kernel/Hardware/MixedAcceptance -maxdepth 4 -type f -print 2>/dev/null || true
  echo
  echo "# Mixed fixture dirs"
  find compiler/test/CodeGen/VC4Kernel/Hardware/Run -maxdepth 2 -type d -name 'mixed_*' -print 2>/dev/null || true
  find compiler/test/CodeGen/SSAVC4/Hardware/Run -maxdepth 2 -type d -name 'mixed_*' -print 2>/dev/null || true
} > "$OUT_ROOT/grep/mixed_acceptance_files.txt"

# libpi runtime if located in common path or env var.
LIBPI_CANDIDATES=()
if [ -n "${VC4_LIBPI_ROOT:-}" ]; then
  LIBPI_CANDIDATES+=("$VC4_LIBPI_ROOT")
fi
LIBPI_CANDIDATES+=("$HOME/Downloads/cs240lx-25spr/libpi")
LIBPI_CANDIDATES+=("$HOME/Downloads/libpi")
LIBPI_CANDIDATES+=("../cs240lx-25spr/libpi")

{
  echo "# libpi candidates"
  for p in "${LIBPI_CANDIDATES[@]}"; do
    echo "$p"
  done
} > "$OUT_ROOT/libpi/candidates.txt"

for p in "${LIBPI_CANDIDATES[@]}"; do
  if [ -d "$p" ]; then
    mkdir -p "$OUT_ROOT/libpi/source"
    copy_if_exists "$p/include/vc4_runtime.h" "$OUT_ROOT/libpi/source/include/vc4_runtime.h"
    copy_if_exists "$p/src/vc4_runtime.c" "$OUT_ROOT/libpi/source/src/vc4_runtime.c"
    copy_if_exists "$p/Makefile" "$OUT_ROOT/libpi/source/Makefile"
    if git -C "$p" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git -C "$p" status --short --branch > "$OUT_ROOT/libpi/git_status.txt" || true
      git -C "$p" log --oneline --decorate -80 > "$OUT_ROOT/libpi/git_log.txt" || true
      git -C "$p" diff --stat > "$OUT_ROOT/libpi/git_diff_stat.txt" || true
      git -C "$p" diff > "$OUT_ROOT/libpi/git_diff.patch" || true
    fi
    echo "$p" > "$OUT_ROOT/libpi/selected_path.txt"
    break
  fi
done

# Remove accidental build/generated bulk
find "$OUT_ROOT" -type d \( -name build -o -name candidate_work -o -name kernels -o -name objs -o -name lowered -o -name hardware \) -prune -exec rm -rf {} + 2>/dev/null || true

# Final inventory
find "$OUT_ROOT" -type f | sort > "$OUT_ROOT/metadata/file_list.txt"
du -sh "$OUT_ROOT" > "$OUT_ROOT/metadata/size.txt" || true

python3 - <<'PY' "$OUT_ROOT" "$ZIP_PATH"
import sys, zipfile, pathlib, hashlib
root = pathlib.Path(sys.argv[1])
zip_path = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(root.parent))
h = hashlib.sha256(zip_path.read_bytes()).hexdigest()
(root / "metadata" / "zip_sha256.txt").write_text(h + "\n")
print(f"CONTEXT_ZIP={zip_path}")
print(f"CONTEXT_SHA256={h}")
PY
