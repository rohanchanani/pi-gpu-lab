#ifndef VC4KERNEL_OPS_TD
#define VC4KERNEL_OPS_TD

include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.td"
include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.td"
include "vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VC4Kernel_I1 : Type<CPred<"$_self.isInteger(1)">, "i1">;
def VC4Kernel_I32 : Type<CPred<"$_self.isSignlessInteger(32)">, "i32">;
def VC4Kernel_F32 : Type<CPred<"$_self.isF32()">, "f32">;
def VC4Kernel_Scalar : AnyTypeOf<[VC4Kernel_I1, VC4Kernel_I32, VC4Kernel_F32]>;
def VC4Kernel_Vector16I32 : Type<CPred<
    "::mlir::vc4kernel::isVC4KernelVector16I32Type($_self)">,
    "vector<16xi32>">;
def VC4Kernel_Vector16F32 : Type<CPred<
    "::mlir::vc4kernel::isVC4KernelVector16F32Type($_self)">,
    "vector<16xf32>">;
def VC4Kernel_Vector16Data
    : AnyTypeOf<[VC4Kernel_Vector16I32, VC4Kernel_Vector16F32]>;

class VC4Kernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<VC4Kernel_Dialect, mnemonic, traits>;

class VC4Kernel_PureOp<string mnemonic, list<Trait> traits = []>
    : VC4Kernel_Op<mnemonic, [Pure] # traits>;

def VC4Kernel_KernelOp : VC4Kernel_Op<"kernel", [
    IsolatedFromAbove, Symbol
  ]> {
  let summary = "strict VC4 target-kernel planning operation";
  let arguments = (ins
    AnyAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    AnyAttr:$public_name,
    AnyAttr:$schedule_mode,
    AnyAttr:$arg_attrs,
    AnyAttr:$resource
  );
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def VC4Kernel_ReturnOp : VC4Kernel_Op<"return", [Terminator]> {
  let summary = "vc4kernel kernel terminator";
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VC4Kernel_ProgramIdOp : VC4Kernel_PureOp<"program_id"> {
  let results = (outs VC4Kernel_I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_BlockIdOp : VC4Kernel_PureOp<"block_id"> {
  let results = (outs VC4Kernel_I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_WarpIdOp : VC4Kernel_PureOp<"warp_id"> {
  let results = (outs VC4Kernel_I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_LaneIdOp : VC4Kernel_PureOp<"lane_id"> {
  let results = (outs VC4Kernel_I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_LaneRangeOp : VC4Kernel_PureOp<"lane_range"> {
  let results = (outs VC4Kernel_Vector16I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_PredFullOp : VC4Kernel_PureOp<"pred.full"> {
  let results = (outs VC4Kernel_PredType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_PredEmptyOp : VC4Kernel_PureOp<"pred.empty"> {
  let results = (outs VC4Kernel_PredType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_PredTailOp : VC4Kernel_PureOp<"pred.tail"> {
  let arguments = (ins VC4Kernel_I32:$base, VC4Kernel_I32:$limit);
  let results = (outs VC4Kernel_PredType:$result);
  let assemblyFormat = "$base `,` $limit attr-dict `:` type($base) `,` type($limit) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_PredRectOp : VC4Kernel_PureOp<"pred.rect"> {
  let arguments = (ins VC4Kernel_I32:$row, VC4Kernel_I32:$rows,
                       VC4Kernel_I32:$col_base, VC4Kernel_I32:$cols);
  let results = (outs VC4Kernel_PredType:$result);
  let assemblyFormat = "$row `,` $rows `,` $col_base `,` $cols attr-dict `:` type($row) `,` type($rows) `,` type($col_base) `,` type($cols) `->` type($result)";
  let hasVerifier = 1;
}

class VC4Kernel_PredBinaryOp<string mnemonic>
    : VC4Kernel_PureOp<mnemonic> {
  let arguments = (ins VC4Kernel_PredType:$lhs, VC4Kernel_PredType:$rhs);
  let results = (outs VC4Kernel_PredType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_PredAndOp : VC4Kernel_PredBinaryOp<"pred.and">;
def VC4Kernel_PredOrOp : VC4Kernel_PredBinaryOp<"pred.or">;

def VC4Kernel_PredNotOp : VC4Kernel_PureOp<"pred.not"> {
  let arguments = (ins VC4Kernel_PredType:$input);
  let results = (outs VC4Kernel_PredType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

class VC4Kernel_PredScalarOp<string mnemonic> : VC4Kernel_PureOp<mnemonic> {
  let arguments = (ins VC4Kernel_PredType:$input);
  let results = (outs VC4Kernel_I1:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_PredAnyOp : VC4Kernel_PredScalarOp<"pred.any">;
def VC4Kernel_PredAllOp : VC4Kernel_PredScalarOp<"pred.all">;

def VC4Kernel_SplatOp : VC4Kernel_PureOp<"splat"> {
  let arguments = (ins AnyType:$input);
  let results = (outs VC4Kernel_Vector16Data:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

class VC4Kernel_FragmentBinaryOp<string mnemonic>
    : VC4Kernel_PureOp<mnemonic> {
  let arguments = (ins VC4Kernel_Vector16Data:$lhs,
                       VC4Kernel_Vector16Data:$rhs);
  let results = (outs VC4Kernel_Vector16Data:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_FragmentAddOp : VC4Kernel_FragmentBinaryOp<"fragment_add">;
def VC4Kernel_FragmentSubOp : VC4Kernel_FragmentBinaryOp<"fragment_sub">;
def VC4Kernel_FragmentMulOp : VC4Kernel_FragmentBinaryOp<"fragment_mul">;

def VC4Kernel_FragmentShlOp : VC4Kernel_PureOp<"fragment_shl"> {
  let arguments = (ins VC4Kernel_Vector16I32:$input, VC4Kernel_I32:$amount);
  let results = (outs VC4Kernel_Vector16I32:$result);
  let assemblyFormat = "$input `,` $amount attr-dict `:` type($input) `,` type($amount) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_FragmentCmpOp : VC4Kernel_PureOp<"fragment_cmp"> {
  let arguments = (ins VC4Kernel_Vector16I32:$lhs,
                       VC4Kernel_Vector16I32:$rhs,
                       VC4Kernel_CmpPredicateAttr:$predicate);
  let results = (outs VC4Kernel_PredType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_FragmentSelectOp : VC4Kernel_PureOp<"fragment_select"> {
  let arguments = (ins VC4Kernel_PredType:$pred,
                       VC4Kernel_Vector16Data:$true_value,
                       VC4Kernel_Vector16Data:$false_value);
  let results = (outs VC4Kernel_Vector16Data:$result);
  let assemblyFormat = "$pred `,` $true_value `,` $false_value attr-dict `:` type($pred) `,` type($true_value) `,` type($false_value) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_FragmentRotateOp : VC4Kernel_PureOp<"fragment_rotate"> {
  let arguments = (ins VC4Kernel_Vector16Data:$input, I32Attr:$amount);
  let results = (outs VC4Kernel_Vector16Data:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_FragmentReduceOp : VC4Kernel_PureOp<"fragment_reduce"> {
  let arguments = (ins VC4Kernel_Vector16Data:$input,
                       VC4Kernel_PredType:$pred,
                       VC4Kernel_ReduceKindAttr:$kind);
  let results = (outs VC4Kernel_Vector16Data:$result);
  let assemblyFormat = "$input `,` $pred attr-dict `:` type($input) `,` type($pred) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_TMULoadFragmentOp : VC4Kernel_Op<"tmu_load_fragment", [
    MemoryEffects<[MemRead]>
  ]> {
  let arguments = (ins VC4Kernel_I32:$base,
                       VC4Kernel_Vector16I32:$byte_offsets,
                       VC4Kernel_PredType:$pred);
  let results = (outs VC4Kernel_Vector16Data:$result);
  let assemblyFormat = "$base `,` $byte_offsets `,` $pred attr-dict `:` type($base) `,` type($byte_offsets) `,` type($pred) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_VDWStoreFragmentOp : VC4Kernel_Op<"vdw_store_fragment", [
    MemoryEffects<[MemWrite]>
  ]> {
  let arguments = (ins VC4Kernel_I32:$base,
                       VC4Kernel_Vector16I32:$byte_offsets,
                       VC4Kernel_Vector16Data:$value,
                       VC4Kernel_PredType:$pred);
  let assemblyFormat = "$base `,` $byte_offsets `,` $value `,` $pred attr-dict `:` type($base) `,` type($byte_offsets) `,` type($value) `,` type($pred)";
  let hasVerifier = 1;
}

def VC4Kernel_VPMAllocOp : VC4Kernel_Op<"vpm_alloc", [
    MemoryEffects<[MemAlloc]>
  ]> {
  let arguments = (ins I32Attr:$rows, I32Attr:$elem_bytes);
  let results = (outs VC4Kernel_VPMTileType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_VPMWriteFragmentOp : VC4Kernel_Op<"vpm_write_fragment", [
    MemoryEffects<[MemWrite]>
  ]> {
  let arguments = (ins VC4Kernel_VPMTileType:$tile,
                       VC4Kernel_I32:$row,
                       VC4Kernel_Vector16Data:$value,
                       VC4Kernel_PredType:$pred,
                       VC4Kernel_VPMOrientationAttr:$orientation);
  let assemblyFormat = "$tile `,` $row `,` $value `,` $pred attr-dict `:` type($tile) `,` type($row) `,` type($value) `,` type($pred)";
  let hasVerifier = 1;
}

def VC4Kernel_VPMReadFragmentOp : VC4Kernel_Op<"vpm_read_fragment", [
    MemoryEffects<[MemRead]>
  ]> {
  let arguments = (ins VC4Kernel_VPMTileType:$tile,
                       VC4Kernel_I32:$row,
                       VC4Kernel_PredType:$pred,
                       VC4Kernel_VPMOrientationAttr:$orientation);
  let results = (outs VC4Kernel_Vector16Data:$result);
  let assemblyFormat = "$tile `,` $row `,` $pred attr-dict `:` type($tile) `,` type($row) `,` type($pred) `->` type($result)";
  let hasVerifier = 1;
}

def VC4Kernel_VDRLoadToVPMOp : VC4Kernel_Op<"vdr_load_to_vpm", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins VC4Kernel_I32:$base,
                       VC4Kernel_I32:$byte_offset,
                       VC4Kernel_VPMTileType:$tile,
                       VC4Kernel_I32:$dst_row,
                       I32Attr:$rows,
                       I32Attr:$cols,
                       I32Attr:$global_stride_bytes,
                       I32Attr:$elem_bytes);
  let assemblyFormat = "$base `,` $byte_offset `,` $tile `,` $dst_row attr-dict `:` type($base) `,` type($byte_offset) `,` type($tile) `,` type($dst_row)";
  let hasVerifier = 1;
}

def VC4Kernel_VDWStoreVPMFragmentOp : VC4Kernel_Op<"vdw_store_vpm_fragment", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let arguments = (ins VC4Kernel_VPMTileType:$tile,
                       VC4Kernel_I32:$src_row,
                       VC4Kernel_I32:$base,
                       VC4Kernel_I32:$byte_offset,
                       VC4Kernel_PredType:$pred,
                       I32Attr:$elem_bytes);
  let assemblyFormat = "$tile `,` $src_row `,` $base `,` $byte_offset `,` $pred attr-dict `:` type($tile) `,` type($src_row) `,` type($base) `,` type($byte_offset) `,` type($pred)";
  let hasVerifier = 1;
}

def VC4Kernel_BarrierOp : VC4Kernel_Op<"barrier", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // VC4KERNEL_OPS_TD
