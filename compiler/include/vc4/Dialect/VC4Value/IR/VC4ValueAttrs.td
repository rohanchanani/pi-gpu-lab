#ifndef VC4VALUE_ATTRS_TD
#define VC4VALUE_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "vc4/Dialect/VC4Value/IR/VC4ValueDialect.td"

class VC4Value_Attr<string name, string mnemonicName>
    : AttrDef<VC4Value_Dialect, name> {
  let mnemonic = mnemonicName;
}

def VC4Value_GlobalMemorySpaceAttr
    : VC4Value_Attr<"GlobalMemorySpace", "global"> {
  let summary = "VC4 value-layer global memory space marker";
  let description = [{
    `#vc4value.global` marks a public value-kernel memref as logical
    VC4 global/device-addressable memory for the Phase 4 public ABI.
    It is a memory-space attribute only: it does not add vc4value memory
    operations, does not select TMU/VDR/VPM/VDW, and does not imply lowering.
  }];
  let assemblyFormat = "";
}

#endif // VC4VALUE_ATTRS_TD
