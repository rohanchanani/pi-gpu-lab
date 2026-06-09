import os
import shutil

import lit.formats

config.name = "VC4 Triton frontend optional"
config.test_format = lit.formats.ShTest()
config.suffixes = [".test"]
config.excludes = ["Support", "lit.cfg.py"]
config.available_features.add("legacy-python-semantic-lowering")
_parent_test_exec_root = getattr(config, "test_exec_root", None)
config.test_source_root = os.path.dirname(__file__)

_compiler_root = os.path.realpath(os.path.join(config.test_source_root, "../.."))
_repo_root = os.path.dirname(_compiler_root)
_configured_obj_root = getattr(config, "vc4_obj_root", None)
if _configured_obj_root:
  config.test_exec_root = os.path.join(
      str(_configured_obj_root), "test", "TritonFrontend")
elif _parent_test_exec_root:
  config.test_exec_root = os.path.join(
      str(_parent_test_exec_root), "TritonFrontend")
else:
  config.test_exec_root = os.path.join(
      _compiler_root, "build-triton-llvm", "test", "TritonFrontend")
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

_tool_dirs = []
_configured_tools_dir = getattr(config, "vc4_tools_dir", None)
_env_vc4_opt = os.environ.get("VC4_OPT")
_env_vc4_triton_opt = os.environ.get("VC4_TRITON_OPT")
_active_tools_dir = config.environment.get("VC4_ACTIVE_TOOLS_DIR")
_parent_tools_dir = None
_cwd_tools_dir = None
if _parent_test_exec_root:
  _parent_tools_dir = os.path.join(os.path.dirname(str(_parent_test_exec_root)), "bin")
if os.path.basename(os.getcwd()) == "test":
  _cwd_tools_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "bin"))
if _is_executable(_env_vc4_opt):
  _tool_dirs.append(os.path.dirname(_env_vc4_opt))
if _is_executable(_env_vc4_triton_opt):
  _tool_dirs.append(os.path.dirname(_env_vc4_triton_opt))
if _cwd_tools_dir and _is_executable(os.path.join(_cwd_tools_dir, "vc4-opt")):
  _tool_dirs.append(_cwd_tools_dir)
elif _parent_tools_dir and _is_executable(os.path.join(_parent_tools_dir, "vc4-opt")):
  _tool_dirs.append(_parent_tools_dir)
elif _active_tools_dir:
  _tool_dirs.append(_active_tools_dir)
elif _configured_tools_dir:
  _tool_dirs.append(str(_configured_tools_dir))
elif not _tool_dirs:
  _tool_dirs.extend([
      os.path.join(_compiler_root, "build-triton-llvm", "bin"),
      os.path.join(_compiler_root, "build", "bin"),
  ])
  _path_vc4_opt = shutil.which("vc4-opt")
  _path_vc4_triton_opt = shutil.which("vc4-triton-opt")
  if _path_vc4_opt:
    _tool_dirs.append(os.path.dirname(_path_vc4_opt))
  if _path_vc4_triton_opt:
    _tool_dirs.append(os.path.dirname(_path_vc4_triton_opt))

for _tool in ["FileCheck"]:
  _path = shutil.which(_tool)
  if _path:
    _tool_dirs.append(os.path.dirname(_path))

_vc4_opt = None
_vc4_triton_opt = None
for _dir in _tool_dirs:
  _candidate_vc4_opt = os.path.join(_dir, "vc4-opt")
  if _vc4_opt is None and _is_executable(_candidate_vc4_opt):
    _vc4_opt = _candidate_vc4_opt
  _candidate_vc4_triton_opt = os.path.join(_dir, "vc4-triton-opt")
  if _vc4_triton_opt is None and _is_executable(_candidate_vc4_triton_opt):
    _vc4_triton_opt = _candidate_vc4_triton_opt

if _vc4_triton_opt:
  config.available_features.add("vc4-triton-cpp-frontend")
  config.substitutions.append(("%vc4_triton_opt", _vc4_triton_opt))
else:
  config.substitutions.append(("%vc4_triton_opt", "false"))

if _vc4_opt:
  config.substitutions.append(("%vc4_opt", _vc4_opt))
else:
  config.substitutions.append(("%vc4_opt", "false"))

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = os.pathsep.join(_tool_dirs + [_existing_path])
