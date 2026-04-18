// RUN: vc4-opt %s --verify-diagnostics

vc4.module @dma_desc_pair_error {
  vc4.func @bad_pitch() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.dma.desc {kind = #vc4.dma_desc_kind<load>, block_mode = #vc4.dma_block_mode<row_row>, orientation = #vc4.dma_orientation<horizontal>, elem_width = #vc4.dma_elem_width<w16>, mpitch = 8 : i32} : !vc4.dma.desc // expected-error {{requires 'mpitch' and 'vpitch' to be provided together}}
    vc4.return
  }
}

vc4.module @dma_desc_stride_error {
  vc4.func @bad_stride() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.dma.desc {kind = #vc4.dma_desc_kind<store>, block_mode = #vc4.dma_block_mode<packed_rows>, orientation = #vc4.dma_orientation<vertical>, elem_width = #vc4.dma_elem_width<w32>, extended_stride = 64 : i32} : !vc4.dma.desc // expected-error {{'extended_stride' requires the base 'stride' attribute}}
    vc4.return
  }
}

vc4.module @dma_desc_units_depth_error {
  vc4.func @bad_units_depth() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.dma.desc {kind = #vc4.dma_desc_kind<load>, block_mode = #vc4.dma_block_mode<row_row>, orientation = #vc4.dma_orientation<horizontal>, elem_width = #vc4.dma_elem_width<w8>, units = 2 : i32, depth = 3 : i32} : !vc4.dma.desc // expected-error {{must not specify both 'units' and 'depth' in one descriptor}}
    vc4.return
  }
}

vc4.module @dma_desc_start_offset_error {
  vc4.func @bad_offset() attributes {threading = 0 : i32, form = 0 : i32} {
    // expected-error@+1 {{'start_offset' attribute must be in range}}
    %0 = vc4.dma.desc {kind = #vc4.dma_desc_kind<store>, block_mode = #vc4.dma_block_mode<row_row>, orientation = #vc4.dma_orientation<vertical>, elem_width = #vc4.dma_elem_width<w16>, start_offset = 4 : i32} : !vc4.dma.desc
    vc4.return
  }
}

vc4.module @dma_start_base_type_error {
  vc4.func @bad_start(%addr: vector<16xi32>) attributes {threading = 0 : i32, form = 0 : i32} {
    %desc = vc4.dma.desc {kind = #vc4.dma_desc_kind<load>, block_mode = #vc4.dma_block_mode<row_row>, orientation = #vc4.dma_orientation<horizontal>, elem_width = #vc4.dma_elem_width<w32>} : !vc4.dma.desc
    // expected-error@+1 {{base address must be a scalar signless integer or index}}
    %0 = "vc4.dma.start"(%desc, %addr) : (!vc4.dma.desc, vector<16xi32>) -> !vc4.async.token
    vc4.return
  }
}

vc4.module @dma_status_type_error {
  vc4.func @bad_status() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.dma.status <load> : f32 // expected-error {{result type must be a scalar signless integer or index}}
    vc4.return
  }
}

vc4.module @dma_wait_both_error {
  vc4.func @bad_wait(%addr: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %desc = vc4.dma.desc {kind = #vc4.dma_desc_kind<load>, block_mode = #vc4.dma_block_mode<row_row>, orientation = #vc4.dma_orientation<horizontal>, elem_width = #vc4.dma_elem_width<w32>} : !vc4.dma.desc
    %tok = "vc4.dma.start"(%desc, %addr) : (!vc4.dma.desc, i32) -> !vc4.async.token
    "vc4.dma.wait"(%tok) <{kind = #vc4.dma_desc_kind<load>}> : (!vc4.async.token) -> () // expected-error {{requires exactly one of a token operand or a 'kind' attribute}}
    vc4.return
  }
}

vc4.module @dma_wait_missing_error {
  vc4.func @bad_wait_missing() attributes {threading = 0 : i32, form = 0 : i32} {
    "vc4.dma.wait"() : () -> () // expected-error {{requires exactly one of a token operand or a 'kind' attribute}}
    vc4.return
  }
}

vc4.module @dma_wait_token_origin_error {
  vc4.func @bad_wait_token(%addr: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %tok = "vc4.tmu.request"(%addr) <{unit = #vc4.tmu_unit<tmu0>}> : (i32) -> !vc4.async.token
    "vc4.dma.wait"(%tok) : (!vc4.async.token) -> () // expected-error {{token operand must come from vc4.dma.start or be a block argument}}
    vc4.return
  }
}
