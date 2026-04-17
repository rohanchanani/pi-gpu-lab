import os

import lit.formats
import lit.llvm
from lit.llvm.subst import ToolSubst

lit.llvm.initialize(lit_config, config)
llvm_config = lit.llvm.llvm_config

config.name = "VC4"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.vc4_obj_root

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "PATH", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.substitutions.append(("%PATH%", config.environment["PATH"]))

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.vc4_tools_dir, config.llvm_tools_dir]
tools = [
    ToolSubst("FileCheck", unresolved="fatal"),
    ToolSubst("count", unresolved="fatal"),
    ToolSubst("not", unresolved="fatal"),
    ToolSubst("vc4-opt", unresolved="fatal"),
    ToolSubst("vc4-translate", unresolved="fatal"),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
