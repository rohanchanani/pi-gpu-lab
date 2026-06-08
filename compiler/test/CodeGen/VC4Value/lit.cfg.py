import os
import shutil

import lit.formats

config.name = "VC4Value codegen"
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
    str(_vc4_repo_root), "compiler", "build", "test", "CodeGen", "VC4Value")

_tools = []
_tools.append(os.path.join(str(_vc4_repo_root), "compiler", "build", "bin"))
_compiler_root = getattr(config, "vc4_source_root", None)
if _compiler_root:
  _tools.append(os.path.join(str(_compiler_root), "build", "bin"))

for _tool in ["FileCheck"]:
  _path = shutil.which(_tool)
  if _path:
    _tools.append(os.path.dirname(_path))

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = os.pathsep.join(_tools + [_existing_path])
