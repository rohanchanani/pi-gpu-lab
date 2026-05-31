import os
import shutil

import lit.formats

config.name = "VC4Kernel to SSAVC4 conversion"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.excludes = []
config.test_source_root = os.path.dirname(__file__)

_compiler_root = os.path.realpath(
    os.path.join(config.test_source_root, "../../.."))
config.test_exec_root = os.path.join(
    _compiler_root, "build", "test", "Conversion", "VC4KernelToSSAVC4")
os.makedirs(config.test_exec_root, exist_ok=True)

_tools = [os.path.join(_compiler_root, "build", "bin")]
for _tool in ["FileCheck", "not"]:
  _path = shutil.which(_tool)
  if _path:
    _tools.append(os.path.dirname(_path))

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = os.path.pathsep.join(_tools + [_existing_path])
