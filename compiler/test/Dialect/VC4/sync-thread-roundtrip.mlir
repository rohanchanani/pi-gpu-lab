// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @sync_thread_ops {
// CHECK: vc4.func @threadable_main(%[[ADDR:.*]]: i32) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<threadable>} {
// CHECK: vc4.mutex <acquire>
// CHECK: vc4.mutex <release>
// CHECK: vc4.semaphore <acquire> {id = 3 : i32}
// CHECK: vc4.semaphore <release> {id = 9 : i32}
// CHECK: vc4.host_interrupt
// CHECK: vc4.thread_switch <switch>
// CHECK: vc4.thread_switch <last_switch>
// CHECK: %[[DMA_DESC:.*]] = vc4.dma.desc {block_mode = #vc4.dma_block_mode<row_row>, elem_width = #vc4.dma_elem_width<w32>, kind = #vc4.dma_desc_kind<load>, orientation = #vc4.dma_orientation<horizontal>} : !vc4.dma.desc
// CHECK: %[[DMA_TOK:.*]] = "vc4.dma.start"(%[[DMA_DESC]], %[[ADDR]]) : (!vc4.dma.desc, i32) -> !vc4.async.token
// CHECK: %[[TMU_TOK:.*]] = "vc4.tmu.request"(%[[ADDR]]) <{unit = #vc4.tmu_unit<tmu0>}> : (i32) -> !vc4.async.token
// CHECK: vc4.async.wait %[[DMA_TOK]], %[[TMU_TOK]] : !vc4.async.token, !vc4.async.token
// CHECK: vc4.program_end
// CHECK: vc4.return
// CHECK: vc4.func @end_only() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
// CHECK: vc4.program_end
// CHECK: vc4.return

vc4.module @sync_thread_ops {
  vc4.func @threadable_main(%addr: i32) attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<threadable>} {
    vc4.mutex <acquire>
    vc4.mutex <release>
    vc4.semaphore <acquire> {id = 3 : i32}
    vc4.semaphore <release> {id = 9 : i32}
    vc4.host_interrupt
    vc4.thread_switch <switch>
    vc4.thread_switch <last_switch>

    %dma_desc = vc4.dma.desc {
      kind = #vc4.dma_desc_kind<load>,
      block_mode = #vc4.dma_block_mode<row_row>,
      orientation = #vc4.dma_orientation<horizontal>,
      elem_width = #vc4.dma_elem_width<w32>
    } : !vc4.dma.desc
    %dma_tok = "vc4.dma.start"(%dma_desc, %addr) : (!vc4.dma.desc, i32) -> !vc4.async.token
    %tmu_tok = "vc4.tmu.request"(%addr) <{unit = #vc4.tmu_unit<tmu0>}> : (i32) -> !vc4.async.token
    vc4.async.wait %dma_tok, %tmu_tok : !vc4.async.token, !vc4.async.token
    vc4.program_end
    vc4.return
  }

  vc4.func @end_only() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    vc4.program_end
    vc4.return
  }
}
