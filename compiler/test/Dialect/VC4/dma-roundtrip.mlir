// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @dma_ops {
// CHECK: vc4.func @main(%[[ADDR:.*]]: i32) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
// CHECK: %[[LOAD_DESC:.*]] = vc4.dma.desc {block_mode = #vc4.dma_block_mode<row_row>, depth = 4 : i32, elem_width = #vc4.dma_elem_width<w16>, kind = #vc4.dma_desc_kind<load>, mpitch = 8 : i32, nrows = 2 : i32, orientation = #vc4.dma_orientation<horizontal>, rowlen = 16 : i32, start_offset = 1 : i32, stride = 32 : i32, vpitch = 2 : i32, vpm_base = 4 : i32} : !vc4.dma.desc
// CHECK: %[[STORE_DESC:.*]] = vc4.dma.desc {block_mode = #vc4.dma_block_mode<packed_rows>, elem_width = #vc4.dma_elem_width<w32>, extended_stride = 64 : i32, kind = #vc4.dma_desc_kind<store>, nrows = 1 : i32, orientation = #vc4.dma_orientation<vertical>, rowlen = 8 : i32, stride = 16 : i32, units = 3 : i32, vpm_base = 12 : i32} : !vc4.dma.desc
// CHECK: %[[TOK:.*]] = "vc4.dma.start"(%[[LOAD_DESC]], %[[ADDR]]) : (!vc4.dma.desc, i32) -> !vc4.async.token
// CHECK: "vc4.dma.start"(%[[STORE_DESC]], %[[ADDR]]) : (!vc4.dma.desc, i32) -> ()
// CHECK: %[[STATUS0:.*]] = vc4.dma.status <load> : i32
// CHECK: %[[STATUS1:.*]] = vc4.dma.status <store> : index
// CHECK: "vc4.dma.wait"(%[[TOK]]) : (!vc4.async.token) -> ()
// CHECK: "vc4.dma.wait"() <{kind = #vc4.dma_desc_kind<store>}> : () -> ()

vc4.module @dma_ops {
  vc4.func @main(%addr: i32) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %load_desc = vc4.dma.desc {
      kind = #vc4.dma_desc_kind<load>,
      block_mode = #vc4.dma_block_mode<row_row>,
      orientation = #vc4.dma_orientation<horizontal>,
      elem_width = #vc4.dma_elem_width<w16>,
      start_offset = 1 : i32,
      mpitch = 8 : i32,
      vpitch = 2 : i32,
      nrows = 2 : i32,
      rowlen = 16 : i32,
      depth = 4 : i32,
      vpm_base = 4 : i32,
      stride = 32 : i32
    } : !vc4.dma.desc
    %store_desc = vc4.dma.desc {
      kind = #vc4.dma_desc_kind<store>,
      block_mode = #vc4.dma_block_mode<packed_rows>,
      orientation = #vc4.dma_orientation<vertical>,
      elem_width = #vc4.dma_elem_width<w32>,
      nrows = 1 : i32,
      rowlen = 8 : i32,
      units = 3 : i32,
      vpm_base = 12 : i32,
      stride = 16 : i32,
      extended_stride = 64 : i32
    } : !vc4.dma.desc

    %tok = "vc4.dma.start"(%load_desc, %addr) : (!vc4.dma.desc, i32) -> !vc4.async.token
    "vc4.dma.start"(%store_desc, %addr) : (!vc4.dma.desc, i32) -> ()
    %status0 = vc4.dma.status <load> : i32
    %status1 = vc4.dma.status <store> : index
    "vc4.dma.wait"(%tok) : (!vc4.async.token) -> ()
    "vc4.dma.wait"() <{kind = #vc4.dma_desc_kind<store>}> : () -> ()
    vc4.return
  }
}
