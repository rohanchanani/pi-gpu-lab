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

_existing_path = config.environment.get("PATH", os.environ.get("PATH", ""))
config.environment["PATH"] = _existing_path
