// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

func.func @cf_switch_valid(%selector: i32 {vc4value.arg_name = "selector"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  // CHECK: cf.switch
  cf.switch %selector : i32, [
    default: ^default,
    0: ^case0,
    1: ^case1
  ]

^case0:
  cf.br ^exit

^case1:
  cf.br ^exit

^default:
  cf.br ^exit

^exit:
  return
}
