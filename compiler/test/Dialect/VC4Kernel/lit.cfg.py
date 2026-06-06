import os
import shutil

import lit.formats

config.name = "VC4Kernel dialect"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".test"]
config.excludes = []
config.test_source_root = os.path.dirname(__file__)

_compiler_root = os.path.realpath(
    os.path.join(config.test_source_root, "../../.."))
config.test_exec_root = os.path.join(
    _compiler_root, "build", "test", "Dialect", "VC4Kernel")
_tools = [os.path.join(_compiler_root, "build", "bin")]
for _tool in ["FileCheck", "not"]:
  _path = shutil.which(_tool)
  if _path:
    _tools.append(os.path.dirname(_path))

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = os.path.pathsep.join(_tools + [_existing_path])
