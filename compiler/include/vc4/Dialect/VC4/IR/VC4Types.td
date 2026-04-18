#ifndef VC4_TYPES_TD
#define VC4_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "vc4/Dialect/VC4/IR/VC4Base.td"

class VC4_Type<string name, string typeMnemonic>
    : TypeDef<VC4_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def VC4_AsyncTokenType : VC4_Type<"AsyncToken", "async.token"> {
  let summary = "opaque async token type";
}

def VC4_TMUDescType : VC4_Type<"TMUDesc", "tmu.desc"> {
  let summary = "opaque TMU descriptor type";
}

def VC4_VPMDescType : VC4_Type<"VPMDesc", "vpm.desc"> {
  let summary = "opaque VPM descriptor type";
}

def VC4_DMADescType : VC4_Type<"DMADesc", "dma.desc"> {
  let summary = "opaque DMA descriptor type";
}

#endif // VC4_TYPES_TD
