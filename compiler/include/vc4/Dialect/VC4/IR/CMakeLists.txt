set(LLVM_TARGET_DEFINITIONS VC4Dialect.td)
mlir_tablegen(VC4Dialect.h.inc -gen-dialect-decls -dialect=vc4)
mlir_tablegen(VC4Dialect.cpp.inc -gen-dialect-defs -dialect=vc4)
add_mlir_dialect_tablegen_target(VC4DialectIncGen)
