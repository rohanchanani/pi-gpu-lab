module {
  tt.func public @probe_supported(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n_elements: i32) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %base = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base_splat = tt.splat %base : i32 -> tensor<16xi32>
    %offsets = arith.addi %base_splat, %range : tensor<16xi32>
    %bound = tt.splat %n_elements : i32 -> tensor<16xi32>
    %mask = arith.cmpi slt, %offsets, %bound : tensor<16xi32>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %x_addr = tt.addptr %x_splat, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %x = tt.load %x_addr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %y_splat = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %y_addr = tt.addptr %y_splat, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %y = tt.load %y_addr, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %sum = arith.addf %x, %y : tensor<16xf32>
    %out_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out_addr = tt.addptr %out_splat, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %out_addr, %sum, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
  tt.func public @probe_dot() attributes {noinline = false} {
    %a = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %b = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return
  }
}
