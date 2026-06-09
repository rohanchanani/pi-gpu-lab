cmake_minimum_required(VERSION 3.20)

project(VC4Compiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VC4_ENABLE_TRITON_CPP_FRONTEND
  "Build optional Triton C++ TTIR frontend"
  OFF)

set(VC4_TRITON_SOURCE_DIR "" CACHE PATH
  "Triton source tree for the optional C++ TTIR frontend")
set(VC4_TRITON_BUILD_DIR "" CACHE PATH
  "Triton CMake build tree for generated headers and explicit objects")
set(VC4_TRITON_LLVM_PREFIX "" CACHE PATH
  "Triton-pinned LLVM/MLIR installation prefix")
set(VC4_TRITON_CPP_OBJECTS "" CACHE STRING
  "Semicolon-separated explicit Triton object files for the optional frontend")
set(VC4_TRITON_CPP_LIBRARIES "" CACHE STRING
  "Semicolon-separated explicit Triton libraries for the optional frontend")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Some installed MLIR packages, including Triton's prebuilt LLVM/MLIR package,
# export concrete MLIR component targets but not the aggregate monolithic target
# named `MLIR`.  VC4 links against MLIR, so do not provide an empty compatibility
# target here; use the aggregate target when available and otherwise link the
# concrete component targets exported by the installed package.
if(TARGET MLIR)
  set(VC4_MLIR_LINK_LIBS MLIR)
  set(VC4_MLIR_TOOL_LINK_LIBS MLIR)
else()
  set(VC4_MLIR_LINK_LIBS
    MLIRArithDialect
    MLIRBytecodeOpInterface
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRControlFlowDialect
    MLIRControlFlowInterfaces
    MLIRDataLayoutInterfaces
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRRewrite
    MLIRSCFDialect
    MLIRShapedOpInterfaces
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransforms
    MLIRVectorDialect
    MLIRViewLikeInterface
  )
  set(VC4_MLIR_TOOL_LINK_LIBS
    ${VC4_MLIR_LINK_LIBS}
    MLIRMlirOptMain
  )
endif()

if(NOT DEFINED LLVM_EXTERNAL_LIT OR LLVM_EXTERNAL_LIT STREQUAL "")
  find_program(VC4_LIT_PROGRAM
    NAMES llvm-lit lit
    HINTS
      "${LLVM_TOOLS_BINARY_DIR}"
      "${LLVM_BINARY_DIR}/bin"
      "${MLIR_INSTALL_PREFIX}/bin"
    DOC "Path to llvm-lit or lit")
  if(VC4_LIT_PROGRAM)
    set(LLVM_EXTERNAL_LIT "${VC4_LIT_PROGRAM}" CACHE FILEPATH
      "Path to llvm-lit or lit" FORCE)
  endif()
endif()

if(NOT LLVM_EXTERNAL_LIT)
  message(WARNING
    "No lit executable was found. Pass -DLLVM_EXTERNAL_LIT=/path/to/llvm-lit "
    "or ensure llvm-lit/lit is on PATH before running check-vc4.")
endif()

if(LLVM_EXTERNAL_LIT AND EXISTS "${LLVM_EXTERNAL_LIT}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import lit"
    RESULT_VARIABLE VC4_PYTHON_CAN_IMPORT_LIT
    OUTPUT_QUIET
    ERROR_QUIET
  )

  if(NOT VC4_PYTHON_CAN_IMPORT_LIT EQUAL 0)
    file(STRINGS "${LLVM_EXTERNAL_LIT}" VC4_LIT_SHEBANG LIMIT_COUNT 1)
    if(VC4_LIT_SHEBANG MATCHES "^#!([^ ]+)( +([^ ]+))?")
      set(VC4_LIT_PYTHON "${CMAKE_MATCH_1}")
      if(VC4_LIT_PYTHON MATCHES "/env$" AND CMAKE_MATCH_3)
        find_program(VC4_LIT_PYTHON NAMES "${CMAKE_MATCH_3}")
      endif()
      if(VC4_LIT_PYTHON AND EXISTS "${VC4_LIT_PYTHON}")
        set(Python3_EXECUTABLE "${VC4_LIT_PYTHON}" CACHE FILEPATH
          "Python interpreter used for lit and MLIR tooling" FORCE)
        message(STATUS "Using lit-compatible Python interpreter: ${Python3_EXECUTABLE}")
      endif()
    endif()
  endif()
endif()

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)

add_definitions(${LLVM_DEFINITIONS})

set(VC4_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(VC4_BINARY_DIR "${PROJECT_BINARY_DIR}")
set(VC4_TOOLS_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(MLIR_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")

enable_testing()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/vc4-opt)

if(VC4_ENABLE_TRITON_CPP_FRONTEND)
  include("${PROJECT_SOURCE_DIR}/cmake/VC4TritonFrontendDeps.cmake")
  add_subdirectory(lib/Conversion/TritonToVC4Value)
  add_subdirectory(tools/vc4-triton-opt)
endif()

add_subdirectory(test)
add_subdirectory(test/Dialect/SSAVC4)
add_subdirectory(test/Dialect/VC4Kernel)
add_subdirectory(test/Dialect/VC4Value)
add_subdirectory(test/ValueSurface)
add_subdirectory(test/Conversion/VC4ValueToVC4Kernel)
add_subdirectory(test/Conversion/VC4KernelToSSAVC4)
add_subdirectory(test/Conversion/SSAVC4ToVC4)
add_subdirectory(test/CodeGen/SSAVC4)
add_subdirectory(test/CodeGen/VC4Kernel)
add_subdirectory(test/CodeGen/VC4Value)
