import os

import lit.formats

config.name = "SSAVC4 lowering"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.excludes = ["Output"]

_source_root = os.path.dirname(__file__)
_compiler_root = os.path.realpath(os.path.join(_source_root, "../../.."))
_configured_root = os.path.join(
    _compiler_root, "build", "test", "Conversion", "SSAVC4ToVC4")
config.test_source_root = _configured_root
config.test_exec_root = _configured_root
os.makedirs(_configured_root, exist_ok=True)

_tools = [os.path.join(_compiler_root, "build", "bin")]
_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = os.pathsep.join(_tools + [_existing_path])
