// RUN: vc4-opt %s --vc4-test-print-effects -o /dev/null | FileCheck %s

// CHECK: vc4.qpu.sema: Read<Semaphore>, Write<Semaphore>
// CHECK: vc4.uniform.read: Read<UniformStream>
// CHECK: vc4.uniform.seek: Write<UniformStream>
// CHECK: vc4.tmu.request: Write<TMUReq0>, Read<MainMemory>
// CHECK: vc4.tmu.request: Write<TMUReq1>, Read<MainMemory>
// CHECK: vc4.tmu.read: Read<TMURcv0>
// CHECK: vc4.tmu.read: Read<TMURcv1>
// CHECK: vc4.tmu.noswap: Write<V3DSystem>
// CHECK: vc4.sfu.issue: Write<SFU>
// CHECK: vc4.sfu.read: Read<SFU>
// CHECK: vc4.vpm.read: Read<VPMReadFIFO>
// CHECK: vc4.vpm.write: Write<VPMWriteFIFO>
// CHECK: vc4.dma.start: Write<VDR>, Read<MainMemory>
// CHECK: vc4.dma.start: Write<VDW>, Write<MainMemory>
// CHECK: vc4.dma.status: Read<VDW>
// CHECK: vc4.dma.wait: Read<VDR>
// CHECK: vc4.mutex: Read<Mutex>, Write<Mutex>
// CHECK: vc4.semaphore: Read<Semaphore>, Write<Semaphore>
// CHECK: vc4.host_interrupt: Write<HostIRQ>
// CHECK: vc4.enqueue_qpu: Write<QPUScheduler>
// CHECK: vc4.reserve_qpu: Write<QPUScheduler>
// CHECK: vc4.v3d.query: Read<V3DSystem>
// CHECK: vc4.v3d.configure: Write<V3DSystem>

vc4.module @side_effects {
  vc4.func @kernel_entry() attributes {kernel, threading = 0 : i32, form = 0 : i32} {
    vc4.return
  }

  vc4.func @scheduled_main() attributes {threading = 0 : i32, form = 1 : i32} {
    vc4.qpu.sema <release> {
      id = 2 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 0 : i32,
      waddr_mul = 0 : i32
    }
  }

  vc4.func @main(%addr: i32, %coord: f32, %vec: vector<16xi32>, %scalar: f32) attributes {threading = 0 : i32, form = 0 : i32} {
    %tex_desc = vc4.tmu.descriptor {
      mode = #vc4.tmu_mode<texture2d>,
      texture_type = #vc4.texture_type<rgba8888>,
      width = 16 : i32,
      height = 16 : i32
    } : !vc4.tmu.desc
    %read_desc = vc4.vpm.desc {
      kind = #vc4.vpm_desc_kind<read>,
      orientation = #vc4.vpm_orientation<horizontal>,
      lane_mode = #vc4.vpm_lane_mode<packed>,
      elem_width = #vc4.vpm_elem_width<w32>,
      num_vectors = 1 : i32
    } : !vc4.vpm.desc
    %write_desc = vc4.vpm.desc {
      kind = #vc4.vpm_desc_kind<write>,
      orientation = #vc4.vpm_orientation<horizontal>,
      lane_mode = #vc4.vpm_lane_mode<packed>,
      elem_width = #vc4.vpm_elem_width<w32>
    } : !vc4.vpm.desc
    %dma_load = vc4.dma.desc {
      kind = #vc4.dma_desc_kind<load>,
      block_mode = #vc4.dma_block_mode<row_row>,
      orientation = #vc4.dma_orientation<horizontal>,
      elem_width = #vc4.dma_elem_width<w32>
    } : !vc4.dma.desc
    %dma_store = vc4.dma.desc {
      kind = #vc4.dma_desc_kind<store>,
      block_mode = #vc4.dma_block_mode<row_row>,
      orientation = #vc4.dma_orientation<horizontal>,
      elem_width = #vc4.dma_elem_width<w32>
    } : !vc4.dma.desc

    %u = vc4.uniform.read : i32
    vc4.uniform.seek %addr : i32
    %tok0 = "vc4.tmu.request"(%addr) <{unit = #vc4.tmu_unit<tmu0>}> : (i32) -> !vc4.async.token
    "vc4.tmu.request"(%coord, %tex_desc) <{unit = #vc4.tmu_unit<tmu1>}> : (f32, !vc4.tmu.desc) -> ()
    %tmu0 = "vc4.tmu.read"() <{unit = #vc4.tmu_unit<tmu0>, part = #vc4.tmu_read_part<raw32>}> : () -> i32
    %tmu1 = "vc4.tmu.read"() <{unit = #vc4.tmu_unit<tmu1>, part = #vc4.tmu_read_part<raw32>}> : () -> i32
    "vc4.tmu.noswap"() <{disable = true}> : () -> ()
    vc4.sfu.issue <recip> %scalar : f32
    %sfu = vc4.sfu.read : f32
    %vpm = vc4.vpm.read %read_desc : (!vc4.vpm.desc) -> vector<16xi32>
    vc4.vpm.write %write_desc, %vec : (!vc4.vpm.desc, vector<16xi32>) -> ()
    %dma_tok = "vc4.dma.start"(%dma_load, %addr) : (!vc4.dma.desc, i32) -> !vc4.async.token
    "vc4.dma.start"(%dma_store, %addr) : (!vc4.dma.desc, i32) -> ()
    %status = vc4.dma.status <store> : i32
    "vc4.dma.wait"(%dma_tok) : (!vc4.async.token) -> ()
    vc4.mutex <acquire>
    vc4.semaphore <release> {id = 1 : i32}
    vc4.host_interrupt
    %qpu_tok = "vc4.enqueue_qpu"(%addr, %addr) <{entry = @kernel_entry}> : (i32, i32) -> !vc4.async.token
    vc4.reserve_qpu {mask = 1 : i32}
    %ident = "vc4.v3d.query"() <{kind = #vc4.v3d_query_kind<ident>}> : () -> i32
    "vc4.v3d.configure"(%addr) <{kind = #vc4.v3d_configure_kind<cache_control>}> : (i32) -> ()
    vc4.return
  }
}
