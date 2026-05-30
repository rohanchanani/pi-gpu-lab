#ifndef VC4KERNEL_DIALECT_TD
#define VC4KERNEL_DIALECT_TD

include "mlir/IR/OpBase.td"

def VC4Kernel_Dialect : Dialect {
  let name = "vc4kernel";
  let cppNamespace = "::mlir::vc4kernel";
  let summary = "VideoCore IV target kernel planning dialect";
  let description = [{
    VC4Kernel is the strict VC4 target-kernel planning layer above SSAVC4.
    It contains 16-lane fragments, structured VC4 predicates, explicit
    TMU/VDR/VPM/VDW memory path choices, logical launch identity, VPM
    resources, barriers, and scalar CFG.
  }];
  let usePropertiesForAttributes = 0;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // VC4KERNEL_DIALECT_TD
