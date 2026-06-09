import os
import shutil

import lit.formats

config.name = "Triton to VC4Value conversion"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".test"]
config.excludes = ["Inputs", "Support", "Output", "CMakeLists.txt", "lit.cfg.py"]
config.test_source_root = os.path.dirname(__file__)

_parent_test_exec_root = getattr(config, "test_exec_root", None)
_configured_obj_root = getattr(config, "vc4_obj_root", None)
if _configured_obj_root:
  config.test_exec_root = os.path.join(
      str(_configured_obj_root), "test", "Conversion", "TritonToVC4Value")
elif _parent_test_exec_root:
  config.test_exec_root = os.path.join(
      str(_parent_test_exec_root), "Conversion", "TritonToVC4Value")
else:
  config.test_exec_root = os.path.join(
      os.path.realpath(os.path.join(config.test_source_root, "../../..")),
      "build-triton-llvm", "test", "Conversion", "TritonToVC4Value")
os.makedirs(config.test_exec_root, exist_ok=True)


def _find_repo_root():
  configured = getattr(config, "vc4_repo_root", None)
  if configured:
    return str(configured)
  candidate = os.path.abspath(config.test_source_root)
  while True:
    if os.path.isdir(os.path.join(candidate, "compiler", "test")):
      return candidate
    parent = os.path.dirname(candidate)
    if parent == candidate:
      raise RuntimeError("could not find VC4 repository root")
    candidate = parent


def _is_executable(path):
  return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


_repo_root = _find_repo_root()
_compiler_root = os.path.join(_repo_root, "compiler")
config.substitutions.append(("%vc4_repo_root", _repo_root))

_tool_dirs = []
for _env_name in ("VC4_OPT", "VC4_TRITON_OPT"):
  _path = os.environ.get(_env_name)
  if _is_executable(_path):
    _tool_dirs.append(os.path.dirname(_path))

_active_tools_dir = config.environment.get("VC4_ACTIVE_TOOLS_DIR")
_configured_tools_dir = getattr(config, "vc4_tools_dir", None)
_parent_tools_dir = None
if _parent_test_exec_root:
  _parent_tools_dir = os.path.join(os.path.dirname(str(_parent_test_exec_root)), "bin")
_cwd_tools_dir = None
if os.path.basename(os.getcwd()) == "test":
  _cwd_tools_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "bin"))

if _cwd_tools_dir and _is_executable(os.path.join(_cwd_tools_dir, "vc4-opt")):
  _tool_dirs.append(_cwd_tools_dir)
elif _active_tools_dir:
  _tool_dirs.append(str(_active_tools_dir))
elif _configured_tools_dir:
  _tool_dirs.append(str(_configured_tools_dir))
elif _parent_tools_dir and _is_executable(os.path.join(_parent_tools_dir, "vc4-opt")):
  _tool_dirs.append(_parent_tools_dir)

if not _tool_dirs:
  _tool_dirs.extend([
      os.path.join(_compiler_root, "build-triton-llvm", "bin"),
      os.path.join(_compiler_root, "build", "bin"),
  ])
  for _tool in ("vc4-opt", "vc4-triton-opt"):
    _path = shutil.which(_tool)
    if _path:
      _tool_dirs.append(os.path.dirname(_path))

for _tool in ("FileCheck", "not"):
  _path = shutil.which(_tool)
  if _path:
    _tool_dirs.append(os.path.dirname(_path))

_vc4_opt = None
_vc4_triton_opt = None
for _dir in _tool_dirs:
  if _vc4_opt is None:
    _candidate = os.path.join(_dir, "vc4-opt")
    if _is_executable(_candidate):
      _vc4_opt = _candidate
  if _vc4_triton_opt is None:
    _candidate = os.path.join(_dir, "vc4-triton-opt")
    if _is_executable(_candidate):
      _vc4_triton_opt = _candidate

if _vc4_triton_opt and _vc4_opt:
  config.available_features.add("vc4-has-triton-cpp-frontend")
  config.available_features.add("vc4-triton-cpp-frontend")
  config.substitutions.append(("%vc4_triton_opt", _vc4_triton_opt))
else:
  config.substitutions.append(("%vc4_triton_opt", "false"))

if _vc4_opt:
  config.substitutions.append(("%vc4_opt_triton", _vc4_opt))
  config.substitutions.append(("%vc4_opt", _vc4_opt))
else:
  config.substitutions.append(("%vc4_opt_triton", "false"))
  config.substitutions.append(("%vc4_opt", "false"))

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = os.pathsep.join(_tool_dirs + [_existing_path])
