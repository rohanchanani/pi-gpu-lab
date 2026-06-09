import os
import shutil

import lit.formats

config.name = "VC4Value to VC4Kernel conversion"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.excludes = []
config.test_source_root = os.path.dirname(__file__)

_compiler_root = os.path.realpath(
    os.path.join(config.test_source_root, "../../.."))
_repo_root = os.path.realpath(os.path.join(_compiler_root, ".."))
config.test_exec_root = os.path.join(
    _compiler_root, "build", "test", "Conversion", "VC4ValueToVC4Kernel")
os.makedirs(config.test_exec_root, exist_ok=True)
config.substitutions.append(("%vc4_repo_root", _repo_root))

_tools = [os.path.join(_compiler_root, "build", "bin")]
for _tool in ["FileCheck", "not"]:
  _path = shutil.which(_tool)
  if _path:
    _tools.append(os.path.dirname(_path))

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = os.path.pathsep.join(_tools + [_existing_path])
