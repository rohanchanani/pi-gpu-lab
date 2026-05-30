#ifndef VC4KERNEL_TYPES_TD
#define VC4KERNEL_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.td"

class VC4Kernel_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<VC4Kernel_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def VC4Kernel_PredType : VC4Kernel_Type<"Pred", "pred"> {
  let summary = "planned 16-lane VC4 predicate";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
  let genVerifyDecl = 1;
}

def VC4Kernel_VPMTileType : VC4Kernel_Type<"VPMTile", "vpm_tile"> {
  let summary = "opaque statically planned VPM tile handle";
  let assemblyFormat = "";
}

#endif // VC4KERNEL_TYPES_TD
