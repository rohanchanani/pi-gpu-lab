import os

import lit.formats

config.name = "VC4 Triton frontend optional"
config.test_format = lit.formats.ShTest()
config.suffixes = [".test"]
config.excludes = ["Support", "lit.cfg.py"]
config.test_source_root = os.path.dirname(__file__)

_compiler_root = os.path.realpath(os.path.join(config.test_source_root, "../.."))
_repo_root = os.path.dirname(_compiler_root)
config.test_exec_root = os.path.join(_compiler_root, "build", "test", "TritonFrontend")
config.substitutions.append(("%vc4_repo_root", _repo_root))

def _is_executable(path):
  return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _candidate_triton_python():
  for _name in ("VC4_TRITON_PYTHON", "TRITON_PYTHON"):
    _path = os.environ.get(_name)
    if _is_executable(_path):
      return _path
  _fallback = os.path.join(_repo_root, ".vc4_auto", "triton_phase6_venv", "bin", "python")
  if _is_executable(_fallback):
    return _fallback
  return None


def _has_pinned_triton_python(path):
  if not path:
    return False
  import subprocess

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

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = _existing_path
