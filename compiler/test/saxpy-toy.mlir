module {
  func.func @saxpy_vec16(%x: memref<16xf32>, %y: memref<16xf32>, %a: f32) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %vx = vector.transfer_read %x[%c0], %cst
      : memref<16xf32>, vector<16xf32>
    %vy = vector.transfer_read %y[%c0], %cst
      : memref<16xf32>, vector<16xf32>
    %va = vector.splat %a : vector<16xf32>
    %scaled = arith.mulf %va, %vx : vector<16xf32>
    %sum = arith.addf %scaled, %vy : vector<16xf32>
    vector.transfer_write %sum, %y[%c0] : vector<16xf32>, memref<16xf32>
    return
  }
}
