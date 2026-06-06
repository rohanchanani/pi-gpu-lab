// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @scalar_i32_arith_roundtrip(%x : i32, %y : i32) attributes {
    public_name = "scalar_i32_arith_roundtrip",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c1 = arith.constant 1 : i32
    // CHECK: arith.addi
    %add = arith.addi %x, %y : i32
    // CHECK: arith.subi
    %sub = arith.subi %x, %y : i32
    // CHECK: arith.muli
    %mul = arith.muli %x, %y : i32
    // CHECK: arith.shli
    %shl = arith.shli %x, %c1 : i32
    // CHECK: arith.shrui
    %shrui = arith.shrui %x, %c1 : i32
    // CHECK: arith.shrsi
    %shrsi = arith.shrsi %x, %c1 : i32
    // CHECK: arith.andi
    %and = arith.andi %x, %y : i32
    // CHECK: arith.ori
    %or = arith.ori %x, %y : i32
    // CHECK: arith.xori
    %xor = arith.xori %x, %y : i32
    // CHECK: arith.minsi
    %minsi = arith.minsi %x, %y : i32
    // CHECK: arith.maxsi
    %maxsi = arith.maxsi %x, %y : i32
    // CHECK: arith.minui
    %minui = arith.minui %x, %y : i32
    // CHECK: arith.maxui
    %maxui = arith.maxui %x, %y : i32
    // CHECK: arith.cmpi eq
    %eq = arith.cmpi eq, %x, %y : i32
    // CHECK: arith.cmpi ne
    %ne = arith.cmpi ne, %x, %y : i32
    // CHECK: arith.cmpi slt
    %slt = arith.cmpi slt, %x, %y : i32
    // CHECK: arith.cmpi sle
    %sle = arith.cmpi sle, %x, %y : i32
    // CHECK: arith.cmpi sgt
    %sgt = arith.cmpi sgt, %x, %y : i32
    // CHECK: arith.cmpi sge
    %sge = arith.cmpi sge, %x, %y : i32
    // CHECK: arith.cmpi ult
    %ult = arith.cmpi ult, %x, %y : i32
    // CHECK: arith.cmpi ule
    %ule = arith.cmpi ule, %x, %y : i32
    // CHECK: arith.cmpi ugt
    %ugt = arith.cmpi ugt, %x, %y : i32
    // CHECK: arith.cmpi uge
    %uge = arith.cmpi uge, %x, %y : i32
    %selected = arith.select %eq, %add, %sub : i32
    %combined0 = arith.addi %selected, %mul : i32
    %combined1 = arith.addi %combined0, %shl : i32
    %combined2 = arith.addi %combined1, %shrui : i32
    %combined3 = arith.addi %combined2, %shrsi : i32
    %combined4 = arith.addi %combined3, %and : i32
    %combined5 = arith.addi %combined4, %or : i32
    %combined6 = arith.addi %combined5, %xor : i32
    %combined7 = arith.addi %combined6, %minsi : i32
    %combined8 = arith.addi %combined7, %maxsi : i32
    %combined9 = arith.addi %combined8, %minui : i32
    %combined10 = arith.addi %combined9, %maxui : i32
    %cond0 = arith.select %ne, %combined10, %x : i32
    %cond1 = arith.select %slt, %cond0, %x : i32
    %cond2 = arith.select %sle, %cond1, %x : i32
    %cond3 = arith.select %sgt, %cond2, %x : i32
    %cond4 = arith.select %sge, %cond3, %x : i32
    %cond5 = arith.select %ult, %cond4, %x : i32
    %cond6 = arith.select %ule, %cond5, %x : i32
    %cond7 = arith.select %ugt, %cond6, %x : i32
    %cond8 = arith.select %uge, %cond7, %x : i32
    vc4kernel.return
  }
}
