#ifndef VC4_STRUCTURED_OPS_TD
#define VC4_STRUCTURED_OPS_TD

include "vc4/Dialect/VC4/IR/VC4Base.td"
include "vc4/Dialect/VC4/IR/VC4Enums.td"
include "vc4/Dialect/VC4/IR/VC4Types.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VC4_Op<string mnemonic, list<Trait> traits = []>
    : Op<VC4_Dialect, mnemonic, traits>;

def VC4_ModuleOp : VC4_Op<"module", [
    IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol
  ] # GraphRegionNoTerminator.traits> {
  let summary = "VC4 target module";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let extraClassDeclaration = [{
    static ModuleOp create(Location loc, StringRef name);
  }];
  let hasVerifier = 1;
}

def VC4_FuncOp : VC4_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
  ]> {
  let summary = "VC4 target function";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<UnitAttr>:$kernel,
    OptionalAttr<VC4_ThreadingModeAttr>:$threading,
    OptionalAttr<VC4_FunctionFormAttr>:$form
  );
  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ThreadingMode threading, FunctionForm form,
                         ArrayRef<NamedAttribute> attrs = {});

    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def VC4_ReturnOp : VC4_Op<"return", [
    HasParent<"FuncOp">, ReturnLike, Terminator
  ]> {
  let summary = "VC4 structured return";
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{});
  }]>];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def VC4_BuiltinOp : VC4_Op<"builtin", [Pure, HasParent<"FuncOp">]> {
  let summary = "VC4 builtin materialization";
  let arguments = (ins VC4_BuiltinKindAttr:$kind);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // VC4_STRUCTURED_OPS_TD
