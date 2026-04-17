#include "vc4/Target/VC4Asm/VC4Asm.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "vc4/Dialect/VC4/VC4Dialect.h"
#include "vc4/Target/KernelModel.h"

namespace mlir::vc4 {
namespace {

static std::string getKernelFunctionName(const LoweredKernelArtifactModel &model) {
  return model.kernelName + "_launch";
}

static std::string getShaderSymbol(const LoweredKernelArtifactModel &model) {
  return model.kernelName + "shader";
}

static std::string getLauncherGuard(const LoweredKernelArtifactModel &model) {
  std::string upper = llvm::convertToSnakeFromCamelCase(model.kernelName);
  for (char &ch : upper) {
    if (ch == '-')
      ch = '_';
    else
      ch = static_cast<char>(::toupper(static_cast<unsigned char>(ch)));
  }
  return "VC4_" + upper + "_KERNEL_LAUNCH_H";
}

static FailureOr<std::string> getCType(Type type) {
  if (auto memref = dyn_cast<MemRefType>(type)) {
    if (memref.getRank() == 1 && memref.isDynamicDim(0) &&
        memref.getElementType().isF32())
      return std::string("float *");
  }
  if (type.isF32())
    return std::string("float");
  if (isa<IndexType>(type))
    return std::string("uint32_t");
  return failure();
}

static LogicalResult emitFunctionSignature(const LoweredKernelArtifactModel &model,
                                           llvm::raw_ostream &output,
                                           bool withNames) {
  output << "int " << getKernelFunctionName(model) << "(struct vc4_runtime *rt";
  for (const PublicArgumentInfo &arg : model.publicArguments) {
    FailureOr<std::string> cType = getCType(arg.type);
    if (failed(cType))
      return failure();
    output << ", " << *cType;
    if (withNames)
      output << " arg" << arg.index;
  }
  output << ")";
  return success();
}

static bool isMemRefArg(const PublicArgumentInfo &arg) {
  auto memref = dyn_cast<MemRefType>(arg.type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getElementType().isF32();
}

static unsigned getExtentArgIndex(const LoweredKernelArtifactModel &model) {
  return model.execution.upperBoundUniformSlot;
}

class LauncherEmitter {
public:
  LauncherEmitter(llvm::raw_ostream &output) : output(output) {}

  LogicalResult emitHeader(const LoweredKernelArtifactModel &model) {
    output << "#ifndef " << getLauncherGuard(model) << "\n";
    output << "#define " << getLauncherGuard(model) << "\n\n";
    output << "#include <stdint.h>\n\n";
    output << "struct vc4_runtime;\n\n";
    if (failed(emitFunctionSignature(model, output, true)))
      return failure();
    output << ";\n\n#endif\n";
    return success();
  }

  LogicalResult emitSource(const LoweredKernelArtifactModel &model) {
    unsigned extentArg = getExtentArgIndex(model);
    DenseSet<unsigned> writtenBackSlots;
    for (const BodyOperationInfo &op : model.body) {
      if (op.kind == BodyOperationKind::DmaStore && op.memrefUniformSlot)
        writtenBackSlots.insert(*op.memrefUniformSlot);
    }

    output << "#include \"kernel_launch.h\"\n\n";
    output << "#include <stddef.h>\n";
    output << "#include <stdint.h>\n";
    output << "#include <string.h>\n\n";
    output << "#define GPU_MEM_FLG 0xC\n";
    output << "#define GPU_BASE 0x40000000\n";
    output << "#define NUM_UNIFS " << model.uniformStream.size() << "\n\n";
    output << "extern const uint32_t " << getShaderSymbol(model) << "[];\n";
    output << "extern const size_t " << getShaderSymbol(model) << "_word_count;\n\n";
    output << "uint32_t mem_alloc(uint32_t size, uint32_t align, uint32_t flags);\n";
    output << "uint32_t mem_free(uint32_t handle);\n";
    output << "uint32_t mem_lock(uint32_t handle);\n";
    output << "uint32_t mem_unlock(uint32_t handle);\n";
    output << "unsigned gpu_fft_base_exec_direct(uint32_t code, uint32_t unifs[], int num_qpus);\n";
    output << "uint32_t vc4_runtime_lane_width(void);\n";
    output << "uint32_t vc4_runtime_active_qpus(const struct vc4_runtime *rt);\n\n";
    output << "enum {\n";
    output << "  VC4_RUNTIME_MAX_QPUS = 12,\n";
    output << "};\n\n";
    output << "struct " << model.kernelName << "_launch_state {\n";
    output << "  uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];\n";
    output << "  uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];\n";
    output << "  uint32_t handle;\n";
    output << "  uint32_t payload[];\n";
    output << "};\n\n";
    output << "static uint32_t vc4_pack_f32(float value) {\n";
    output << "  uint32_t bits = 0;\n";
    output << "  memcpy(&bits, &value, sizeof(bits));\n";
    output << "  return bits;\n";
    output << "}\n\n";
    output << "static uint32_t *" << model.kernelName
           << "_code_ptr(volatile struct " << model.kernelName << "_launch_state *state) {\n";
    output << "  return (uint32_t *)state->payload;\n";
    output << "}\n\n";
    output << "static size_t " << model.kernelName << "_code_bytes(void) {\n";
    output << "  return " << getShaderSymbol(model) << "_word_count * sizeof(uint32_t);\n";
    output << "}\n\n";
    output << "static size_t " << model.kernelName << "_payload_offset_bytes(void) {\n";
    output << "  return " << model.kernelName << "_code_bytes();\n";
    output << "}\n\n";

    for (const PublicArgumentInfo &arg : model.publicArguments) {
      if (!isMemRefArg(arg))
        continue;
      output << "static float *" << model.kernelName << "_arg" << arg.index
             << "_ptr(volatile struct " << model.kernelName
             << "_launch_state *state, uint32_t extent) {\n";
      output << "  return (float *)((uint8_t *)state->payload + "
             << model.kernelName << "_payload_offset_bytes()";
      for (const PublicArgumentInfo &prior : model.publicArguments) {
        if (prior.index == arg.index)
          break;
        if (isMemRefArg(prior))
          output << " + (size_t)extent * sizeof(float)";
      }
      output << ");\n";
      output << "}\n\n";
    }

    output << "static size_t " << model.kernelName << "_launch_state_size(uint32_t extent) {\n";
    output << "  return offsetof(struct " << model.kernelName
           << "_launch_state, payload) + " << model.kernelName
           << "_payload_offset_bytes()";
    for (const PublicArgumentInfo &arg : model.publicArguments) {
      if (isMemRefArg(arg))
        output << " + (size_t)extent * sizeof(float)";
    }
    output << ";\n";
    output << "}\n\n";

    if (failed(emitFunctionSignature(model, output, true)))
      return failure();
    output << " {\n";
    output << "  uint32_t activeQpus = vc4_runtime_active_qpus(rt);\n";
    output << "  uint32_t laneWidth = vc4_runtime_lane_width();\n\n";
    output << "  if (!rt)\n";
    output << "    return -1;\n";
    for (const PublicArgumentInfo &arg : model.publicArguments) {
      if (isMemRefArg(arg))
        output << "  if (!arg" << arg.index << ")\n    return -1;\n";
    }
    output << "  if (activeQpus == 0 || activeQpus > VC4_RUNTIME_MAX_QPUS)\n";
    output << "    return -1;\n";
    output << "  (void)laneWidth;\n\n";
    output << "  size_t allocSize = " << model.kernelName << "_launch_state_size(arg"
           << extentArg << ");\n";
    output << "  uint32_t handle = mem_alloc((uint32_t)allocSize, 4096, GPU_MEM_FLG);\n";
    output << "  if (!handle)\n";
    output << "    return -1;\n\n";
    output << "  uint32_t vc = mem_lock(handle);\n";
    output << "  if (!vc) {\n";
    output << "    mem_free(handle);\n";
    output << "    return -1;\n";
    output << "  }\n\n";
    output << "  volatile struct " << model.kernelName
           << "_launch_state *state =\n";
    output << "      (volatile struct " << model.kernelName
           << "_launch_state *)(vc - GPU_BASE);\n";
    output << "  if (!state) {\n";
    output << "    mem_unlock(handle);\n";
    output << "    mem_free(handle);\n";
    output << "    return -1;\n";
    output << "  }\n\n";
    output << "  state->handle = handle;\n";
    output << "  memcpy((void *)" << model.kernelName << "_code_ptr(state), "
           << getShaderSymbol(model) << ", " << model.kernelName
           << "_code_bytes());\n\n";

    for (const PublicArgumentInfo &arg : model.publicArguments) {
      if (!isMemRefArg(arg))
        continue;
      output << "  float *gpuArg" << arg.index << " = " << model.kernelName
             << "_arg" << arg.index << "_ptr(state, arg" << extentArg << ");\n";
      output << "  memcpy(gpuArg" << arg.index << ", arg" << arg.index
             << ", (size_t)arg" << extentArg << " * sizeof(float));\n";
      output << "  uint32_t gpuArg" << arg.index << "Addr = GPU_BASE + (uint32_t)gpuArg"
             << arg.index << ";\n\n";
    }

    output << "  for (uint32_t qpu = 0; qpu < activeQpus; ++qpu) {\n";
    output << "    /* user uniforms first, builtin suffix second */\n";
    for (const UniformStreamEntry &entry : model.uniformStream) {
      output << "    state->unif[qpu][" << entry.slot << "] = ";
      if (entry.sourceKind == UniformSourceKind::Builtin) {
        if (entry.builtinKind == BuiltinKind::qpu_id)
          output << "qpu";
        else if (entry.builtinKind == BuiltinKind::num_qpus)
          output << "activeQpus";
        else
          return failure();
      } else {
        const PublicArgumentInfo &arg = model.publicArguments[entry.publicArgumentIndex];
        if (isMemRefArg(arg))
          output << "gpuArg" << arg.index << "Addr";
        else if (arg.type.isF32())
          output << "vc4_pack_f32(arg" << arg.index << ")";
        else if (isa<IndexType>(arg.type))
          output << "arg" << arg.index;
        else
          return failure();
      }
      output << ";\n";
    }
    output << "    state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu];\n";
    output << "  }\n\n";

    output << "  gpu_fft_base_exec_direct((uint32_t)" << model.kernelName
           << "_code_ptr(state), (uint32_t *)state->unif_ptr, activeQpus);\n";
    for (const PublicArgumentInfo &arg : model.publicArguments) {
      if (!isMemRefArg(arg) || !writtenBackSlots.contains(arg.index))
        continue;
      output << "  memcpy(arg" << arg.index << ", gpuArg" << arg.index
             << ", (size_t)arg" << extentArg << " * sizeof(float));\n";
    }
    output << "\n  mem_unlock(handle);\n";
    output << "  mem_free(handle);\n";
    output << "  return 0;\n";
    output << "}\n";
    return success();
  }

private:
  llvm::raw_ostream &output;
};

} // namespace

LogicalResult emitVC4LauncherHeader(const LoweredKernelArtifactModel &model,
                                    llvm::raw_ostream &output) {
  LauncherEmitter emitter(output);
  return emitter.emitHeader(model);
}

LogicalResult emitVC4LauncherHeader(Operation *op, llvm::raw_ostream &output) {
  FailureOr<LoweredKernelArtifactModel> model = extractLoweredKernelArtifactModel(op);
  if (failed(model))
    return failure();
  return emitVC4LauncherHeader(*model, output);
}

LogicalResult emitVC4LauncherSource(const LoweredKernelArtifactModel &model,
                                    llvm::raw_ostream &output) {
  LauncherEmitter emitter(output);
  return emitter.emitSource(model);
}

LogicalResult emitVC4LauncherSource(Operation *op, llvm::raw_ostream &output) {
  FailureOr<LoweredKernelArtifactModel> model = extractLoweredKernelArtifactModel(op);
  if (failed(model))
    return failure();
  return emitVC4LauncherSource(*model, output);
}

void registerToVC4LauncherHeaderTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-vc4-launcher-h",
      "Translate lowered VC4 MLIR to generated kernel_launch.h for the current supported subset",
      [](Operation *op, llvm::raw_ostream &output) {
        return emitVC4LauncherHeader(op, output);
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<VC4Dialect>();
      });
}

void registerToVC4LauncherSourceTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-vc4-launcher-c",
      "Translate lowered VC4 MLIR to generated kernel_launch.c for the current supported subset",
      [](Operation *op, llvm::raw_ostream &output) {
        return emitVC4LauncherSource(op, output);
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registry.insert<VC4Dialect>();
      });
}

} // namespace mlir::vc4
