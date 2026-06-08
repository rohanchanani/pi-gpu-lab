set(LLVM_TARGET_DEFINITIONS VC4ValueDialect.td)
mlir_tablegen(VC4ValueDialect.h.inc -gen-dialect-decls -dialect=vc4value)
mlir_tablegen(VC4ValueDialect.cpp.inc -gen-dialect-defs -dialect=vc4value)
add_public_tablegen_target(VC4ValueDialectIncGen)
