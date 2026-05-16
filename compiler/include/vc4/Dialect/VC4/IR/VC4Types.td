#ifndef VC4_TYPES_TD
#define VC4_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "vc4/Dialect/VC4/IR/VC4Base.td"

class VC4_Type<string name, string typeMnemonic>
    : TypeDef<VC4_Dialect, name> {
  let mnemonic = typeMnemonic;
}

#endif // VC4_TYPES_TD
