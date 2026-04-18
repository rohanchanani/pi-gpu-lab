import lit.formats
import lit.llvm

config.name = "VC4"
lit.llvm.initialize(lit_config, config)
llvm_config = lit.llvm.llvm_config
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.excludes = ["Inputs", "CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

tool_dirs = [config.vc4_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir]
tools = ["vc4-opt", "FileCheck"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
