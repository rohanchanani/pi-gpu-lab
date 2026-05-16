set(LLVM_TARGET_DEFINITIONS SSAVC4Dialect.td)
mlir_tablegen(SSAVC4Dialect.h.inc -gen-dialect-decls -dialect=ssavc4)
mlir_tablegen(SSAVC4Dialect.cpp.inc -gen-dialect-defs -dialect=ssavc4)
add_public_tablegen_target(SSAVC4DialectIncGen)

set(LLVM_TARGET_DEFINITIONS SSAVC4Ops.td)
mlir_tablegen(SSAVC4Ops.h.inc -gen-op-decls)
mlir_tablegen(SSAVC4Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(SSAVC4OpsIncGen)
