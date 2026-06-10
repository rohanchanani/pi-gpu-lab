#!/usr/bin/env python3
"""Create a pinned local Triton TTIR generator environment.

This tool sets up the optional Python lane used to regenerate real TTIR
snapshots. It does not fetch Triton, clone LLVM, or install an unpinned Triton
wheel. The caller must provide an explicit local Triton checkout and locked LLVM
prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_TRITON_TAG = "v3.7.0"
DEFAULT_TRITON_HEAD = "5f3f125e8f63c24613f1f73b937442864f263f94"
DEFAULT_TRITON_MAJOR_MINOR = "3.7"
DEFAULT_LLVM_HASH = "ac5dc54d509169d387fcfd495d71853d81c46484"
DEFAULT_TARGET = "cuda:80:32"
DEFAULT_OUT = Path(".vc4_auto/ttir_generator")
REQUIREMENTS = Path("tools/requirements-triton-generator.txt")
DEFAULT_SPEC = Path("tools/vc4_ttir_generator_toolchain.json")


def fail(message: str) -> "None":
    print(f"vc4_setup_ttir_generator.py: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if capture:
            if exc.stdout:
                print(exc.stdout, file=sys.stderr, end="")
            if exc.stderr:
                print(exc.stderr, file=sys.stderr, end="")
        fail(f"command failed: {' '.join(cmd)}")


def git_value(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args], capture=True).stdout.strip()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        fail(f"{description} does not exist: {path}")
    return path.resolve()


def load_spec(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_built_extension_dir(triton_source: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = require_path(explicit, "Triton Python build extension directory")
        if (candidate / "libtriton.so").exists():
            return candidate
        if (candidate / "triton" / "_C" / "libtriton.so").exists():
            return candidate / "triton" / "_C"
        fail(f"could not find libtriton.so under {candidate}")

    build_root = triton_source / "build"
    candidates = sorted(build_root.glob("lib.*-cpython-*/triton/_C/libtriton.so"))
    if not candidates:
        fail(
            "could not discover built Triton Python extension. Pass "
            "--triton-python-extension-dir pointing at a directory containing "
            "libtriton.so and libproton.so"
        )
    return candidates[-1].parent.resolve()


def venv_python(out: Path) -> Path:
    if sys.platform == "win32":
        return out / "venv" / "Scripts" / "python.exe"
    return out / "venv" / "bin" / "python"


def site_packages(python: Path) -> Path:
    code = (
        "import sysconfig; "
        "print(sysconfig.get_paths()[\"purelib\"])"
    )
    return Path(run([str(python), "-c", code], capture=True).stdout.strip())


def write_overlay(out: Path, triton_source: Path, extension_dir: Path) -> Path:
    overlay = out / "triton_python_overlay" / "triton"
    c_dir = overlay / "_C"
    c_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(extension_dir / "libtriton.so", c_dir / "libtriton.so")
    libproton = extension_dir / "libproton.so"
    if libproton.exists():
        shutil.copy2(libproton, c_dir / "libproton.so")

    source_pkg = triton_source / "python" / "triton"
    init = overlay / "__init__.py"
    init.write_text(
        "# VC4 TTIR generator overlay: load the pinned Triton Python package\n"
        "# while making the locally built libtriton extension visible before\n"
        "# source-tree stubs.\n"
        "import pathlib\n"
        "_overlay = pathlib.Path(__file__).resolve().parent\n"
        f"_source = pathlib.Path({str(source_pkg)!r})\n"
        "__path__ = [str(_overlay), str(_source)]\n"
        "__file__ = str(_source / \"__init__.py\")\n"
        "_code = (_source / \"__init__.py\").read_text(encoding=\"utf-8\")\n"
        "exec(compile(_code, __file__, \"exec\"), globals(), globals())\n",
        encoding="utf-8",
    )
    return overlay.parent.resolve()


def write_pth(python: Path, overlay_root: Path, triton_source: Path) -> Path:
    site = site_packages(python)
    pth = site / "vc4_ttir_generator.pth"
    pth.write_text(
        f"{overlay_root}\n"
        f"{triton_source / 'python'}\n",
        encoding="utf-8",
    )
    return pth


def write_wrapper(
    repo_root: Path,
    out: Path,
    llvm_prefix: Path,
    triton_source: Path,
) -> Path:
    wrapper = out / "run_vc4_emit_ttir.sh"
    python = venv_python(out)
    text = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT={str(repo_root)!r}
VENV={str(out / "venv")!r}
export TRITON_HOME={str(llvm_prefix.parent.parent.parent)!r}
export LLVM_SYSPATH={str(llvm_prefix)!r}
export TRITON_LIBDEVICE_PATH={str(triton_source / "third_party/nvidia/backend/lib")!r}

exec {str(python)!r} "$ROOT/tools/vc4_emit_ttir.py" "$@"
"""
    wrapper.write_text(text, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def python_probe(python: Path, required_major_minor: str) -> dict[str, str]:
    code = f"""
import inspect
import triton
from triton._C.libtriton import ir, getenv, getenv_bool
import triton._C.libtriton as libtriton
version = getattr(triton, "__version__", "<missing>")
if not str(version).startswith({required_major_minor!r} + "."):
    raise SystemExit(f"unexpected Triton version: {{version}}")
print("triton_version=" + str(version))
print("triton_file=" + inspect.getfile(triton))
print("libtriton_file=" + getattr(libtriton, "__file__", "<unknown>"))
"""
    output = run([str(python), "-c", code], capture=True).stdout
    result: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-source", required=True, type=Path)
    parser.add_argument("--llvm-prefix", required=True, type=Path)
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--triton-python-extension-dir", type=Path)
    parser.add_argument("--spec", default=DEFAULT_SPEC, type=Path)
    parser.add_argument("--expected-triton-tag", default=DEFAULT_TRITON_TAG)
    parser.add_argument("--expected-triton-head", default=DEFAULT_TRITON_HEAD)
    parser.add_argument("--expected-triton-major-minor", default=DEFAULT_TRITON_MAJOR_MINOR)
    parser.add_argument("--expected-llvm-hash", default=DEFAULT_LLVM_HASH)
    parser.add_argument("--requirements", default=REQUIREMENTS, type=Path)
    parser.add_argument("--force", action="store_true", help="remove an existing output directory")
    parser.add_argument(
        "--skip-pip-install",
        action="store_true",
        help="create overlay/wrapper without installing small Python dependencies",
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    triton_spec = spec.get("triton", {})
    llvm_spec = spec.get("llvm", {})
    if args.expected_triton_tag == DEFAULT_TRITON_TAG:
        args.expected_triton_tag = triton_spec.get("tag", args.expected_triton_tag)
    if args.expected_triton_head == DEFAULT_TRITON_HEAD:
        args.expected_triton_head = triton_spec.get("source_head", args.expected_triton_head)
    if args.expected_triton_major_minor == DEFAULT_TRITON_MAJOR_MINOR:
        args.expected_triton_major_minor = triton_spec.get(
            "major_minor", args.expected_triton_major_minor
        )
    if args.expected_llvm_hash == DEFAULT_LLVM_HASH:
        args.expected_llvm_hash = llvm_spec.get("hash", args.expected_llvm_hash)
    if args.requirements == REQUIREMENTS and spec.get("requirements"):
        args.requirements = Path(spec["requirements"])

    repo_root = require_path(args.repo_root, "repo root")
    triton_source = require_path(args.triton_source, "Triton source checkout")
    llvm_prefix = require_path(args.llvm_prefix, "Triton LLVM prefix")
    requirements = require_path(args.requirements, "requirements file")

    llvm_hash_prefix = args.expected_llvm_hash[:8]
    if args.expected_llvm_hash not in str(llvm_prefix) and llvm_hash_prefix not in str(llvm_prefix):
        fail(
            f"LLVM prefix does not contain expected hash {args.expected_llvm_hash} "
            f"or prefix {llvm_hash_prefix}: {llvm_prefix}"
        )

    head = git_value(triton_source, "rev-parse", "HEAD")
    desc = git_value(triton_source, "describe", "--tags", "--always")
    if head != args.expected_triton_head:
        fail(f"Triton checkout HEAD mismatch: expected {args.expected_triton_head}, got {head}")
    if desc != args.expected_triton_tag:
        fail(f"Triton checkout tag mismatch: expected {args.expected_triton_tag}, got {desc}")

    extension_dir = find_built_extension_dir(triton_source, args.triton_python_extension_dir)
    require_path(extension_dir / "libtriton.so", "libtriton extension")

    out = args.out.resolve()
    if out.exists():
        if not args.force:
            fail(f"output directory already exists; pass --force to replace it: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    run([sys.executable, "-m", "venv", str(out / "venv")])
    python = venv_python(out)
    if not args.skip_pip_install:
        run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        run([str(python), "-m", "pip", "install", "-r", str(requirements)])

    overlay_root = write_overlay(out, triton_source, extension_dir)
    pth = write_pth(python, overlay_root, triton_source)
    wrapper = write_wrapper(repo_root.resolve(), out, llvm_prefix, triton_source)
    probe = python_probe(python, args.expected_triton_major_minor)

    metadata = {
        "tool_schema_version": 1,
        "triton_source": str(triton_source),
        "triton_source_head": head,
        "triton_source_tag": desc,
        "triton_version": probe.get("triton_version"),
        "triton_file": probe.get("triton_file"),
        "libtriton_file": probe.get("libtriton_file"),
        "llvm_prefix": str(llvm_prefix),
        "expected_llvm_hash": args.expected_llvm_hash,
        "extension_dir": str(extension_dir),
        "venv_python": str(python),
        "overlay_root": str(overlay_root),
        "pth": str(pth),
        "wrapper": str(wrapper),
        "requirements": str(requirements),
        "spec": str(args.spec),
        "ready_for_triton": "NO",
    }
    metadata_path = out / "generator_env.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("VC4_TTIR_GENERATOR_SETUP=PASS")
    print(f"TRITON_TAG={desc}")
    print(f"TRITON_SOURCE_HEAD={head}")
    print(f"TRITON_VERSION={probe.get('triton_version')}")
    print(f"GENERATOR_WRAPPER={wrapper}")
    print(f"GENERATOR_METADATA={metadata_path}")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
