#ifndef SSAVC4_TYPES_TD
#define SSAVC4_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "vc4/Dialect/SSAVC4/IR/SSAVC4Base.td"

class SSAVC4_Type<string name, string typeMnemonic>
    : TypeDef<SSAVC4_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def SSAVC4_AsyncTokenType : SSAVC4_Type<"AsyncToken", "async.token"> {
  let summary = "ordered async hardware token";
  let description = [{
    Token for explicitly ordering async-like hardware queues before lowering.
  }];
}

def SSAVC4_TMUDescType : SSAVC4_Type<"TMUDesc", "tmu.desc"> {
  let summary = "opaque TMU descriptor";
  let description = [{
    Minimal opaque descriptor for SSAVC4 TMU concepts that are not part of the
    post-cleanup scheduled VC4 dialect.
  }];
}

def SSAVC4_VPMDescType : SSAVC4_Type<"VPMDesc", "vpm.desc"> {
  let summary = "opaque VPM descriptor";
  let description = [{
    Minimal opaque descriptor for SSAVC4 VPM/VDW setup concepts that are not
    part of the post-cleanup scheduled VC4 dialect.
  }];
}

def SSAVC4_FlagsType : SSAVC4_Type<"Flags", "flags"> {
  let summary = "restricted SSA pseudo-value for QPU condition flags";
  let description = [{
    Pseudo-value used by SSAVC4 before lowering materializes transient QPU
    condition flags. Flags are deliberately not ordinary data values.
  }];
}

def SSAVC4_AsyncToken : Type<
  CPred<"::llvm::isa<::mlir::ssavc4::AsyncTokenType>($_self)">,
  "!ssavc4.async.token">;
def SSAVC4_TMUDesc : Type<
  CPred<"::llvm::isa<::mlir::ssavc4::TMUDescType>($_self)">,
  "!ssavc4.tmu.desc">;
def SSAVC4_VPMDesc : Type<
  CPred<"::llvm::isa<::mlir::ssavc4::VPMDescType>($_self)">,
  "!ssavc4.vpm.desc">;
def SSAVC4_Flags : Type<
  CPred<"::llvm::isa<::mlir::ssavc4::FlagsType>($_self)">,
  "!ssavc4.flags">;
def SSAVC4_ValueType : Type<
  CPred<"::mlir::ssavc4::isSSAVC4ValueType($_self)">,
  "SSAVC4 lowerable i32/f32/vector<16x...> value type">;

#endif // SSAVC4_TYPES_TD
