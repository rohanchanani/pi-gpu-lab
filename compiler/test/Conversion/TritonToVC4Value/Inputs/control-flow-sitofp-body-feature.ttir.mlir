module {
  tt.func public @control_flow_sitofp_body_feature(%out_ptr: !tt.ptr<f32>, %n_elements: i32, %flag: i32) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %base = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base_splat = tt.splat %base : i32 -> tensor<16xi32>
    %offsets = arith.addi %base_splat, %range : tensor<16xi32>
    %bound = tt.splat %n_elements : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<16xi32>
    %as_f32 = arith.sitofp %flag : i32 to f32
    %vec = tt.splat %as_f32 : f32 -> tensor<16xf32>
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_addr = tt.addptr %out_splat, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %out_addr, %vec, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
