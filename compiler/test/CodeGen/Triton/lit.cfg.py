import os
import shutil
import subprocess

import lit.formats

config.name = "VC4 Triton codegen"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".test"]
config.excludes = ["Hardware", "Support", "Output"]
config.test_source_root = os.path.dirname(__file__)

_vc4_repo_root = getattr(config, "vc4_repo_root", None)
if _vc4_repo_root is None:
  _candidate = os.path.abspath(config.test_source_root)
  while True:
    if os.path.isdir(os.path.join(_candidate, "compiler", "test")):
      _vc4_repo_root = _candidate
      break
    _parent = os.path.dirname(_candidate)
    if _parent == _candidate:
      raise RuntimeError("could not find VC4 repository root from lit config")
    _candidate = _parent

config.substitutions.append(("%vc4_repo_root", str(_vc4_repo_root)))
config.test_exec_root = os.path.join(
    str(_vc4_repo_root), "compiler", "build", "test", "CodeGen", "Triton")

def _is_executable(path):
  return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _candidate_triton_python():
  for _name in ("VC4_TRITON_PYTHON", "TRITON_PYTHON"):
    _path = os.environ.get(_name)
    if _is_executable(_path):
      return _path
  _fallback = os.path.join(str(_vc4_repo_root), ".vc4_auto", "triton_phase6_venv", "bin", "python")
  if _is_executable(_fallback):
    return _fallback
  return None


def _has_pinned_triton_python(path):
  if not path:
    return False
  _check = (
      "import triton\n"
      "raise SystemExit(0 if getattr(triton, '__version__', '').startswith('3.7.') else 1)\n"
  )
  try:
    _proc = subprocess.run(
        [path, "-c", _check],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
  except OSError:
    return False
  return _proc.returncode == 0


_triton_python = _candidate_triton_python()
if _has_pinned_triton_python(_triton_python):
  config.available_features.add("vc4-has-triton-python")
  config.substitutions.append(("%vc4_triton_python", _triton_python))
else:
  config.substitutions.append(("%vc4_triton_python", "false"))

_tools = [os.path.join(str(_vc4_repo_root), "compiler", "build", "bin")]
_compiler_root = getattr(config, "vc4_source_root", None)
if _compiler_root:
  _tools.append(os.path.join(str(_compiler_root), "build", "bin"))

for _tool in ["FileCheck"]:
  _path = shutil.which(_tool)
  if _path:
    _tools.append(os.path.dirname(_path))

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = os.pathsep.join(_tools + [_existing_path])
