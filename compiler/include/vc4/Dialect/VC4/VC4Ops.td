include "vc4/Dialect/VC4/VC4Dialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"

class VC4_Op<string mnemonic, list<Trait> traits = []> :
    Op<VC4_Dialect, mnemonic, traits>;

def VC4_GetUniformOp : VC4_Op<"get_uniform"> {
  let summary = "Read a kernel uniform by slot";
  let description = [{
    Reads one uniform / kernel parameter and returns it as a typed SSA value.
    The slot number is a compact stand-in for the eventual launch convention.
  }];
  let arguments = (ins I32Attr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$index attr-dict `:` type($result)";
}

def VC4_DmaLoadOp : VC4_Op<"dma_load"> {
  let summary = "Stage data from global memory into VPM";
  let description = [{
    Represents a global-to-VPM DMA transfer from a scalar-element memref view at
    a given global element offset into a selected VPM row.
  }];
  let arguments = (ins AnyMemRef:$source, Index:$globalOffset, Index:$row);
}

def VC4_DmaStoreOp : VC4_Op<"dma_store"> {
  let summary = "Store staged data from VPM back to global memory";
  let description = [{
    Represents a VPM-to-global DMA transfer from a selected VPM row into a
    scalar-element memref view at a given global element offset.
  }];
  let arguments = (ins AnyMemRef:$target, Index:$globalOffset, Index:$row);
}

def VC4_VPMReadOp : VC4_Op<"vpm_read"> {
  let summary = "Read a vector value from a VPM row";
  let description = [{
    Reads a staged vector from a VPM row into SSA form so upstream vector and
    arithmetic dialects can operate on it.
  }];
  let arguments = (ins Index:$row);
  let results = (outs AnyVectorOfAnyRank:$result);
  let assemblyFormat = "$row attr-dict `:` type($row) `->` type($result)";
}

def VC4_VPMWriteOp : VC4_Op<"vpm_write"> {
  let summary = "Write a vector value into a VPM row";
  let description = [{
    Writes a vector SSA value into a VPM row, making the staged result available
    for later DMA store or additional staged-memory operations.
  }];
  let arguments = (ins Index:$row, AnyVectorOfAnyRank:$value);
  let assemblyFormat =
      "$row `,` $value attr-dict `:` type($row) `,` type($value)";
}
