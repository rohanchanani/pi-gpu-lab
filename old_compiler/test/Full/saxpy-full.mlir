gpu.module @kernels {
  gpu.func @saxpy(%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %n: index) kernel {
    %gid = gpu.global_id x
    %inBounds = arith.cmpi ult, %gid, %n : index
    scf.if %inBounds {
      %xval = memref.load %x[%gid] : memref<?xf32>
      %yval = memref.load %y[%gid] : memref<?xf32>
      %mul = arith.mulf %a, %xval : f32
      %sum = arith.addf %mul, %yval : f32
      memref.store %sum, %y[%gid] : memref<?xf32>
    }
    gpu.return
  }
}
