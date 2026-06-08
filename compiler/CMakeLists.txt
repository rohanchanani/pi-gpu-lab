cmake_minimum_required(VERSION 3.20)

project(VC4Compiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
add_subdirectory(test)
add_subdirectory(test/Dialect/SSAVC4)
add_subdirectory(test/Dialect/VC4Kernel)
add_subdirectory(test/Dialect/VC4Value)
add_subdirectory(test/ValueSurface)
add_subdirectory(test/Conversion/VC4KernelToSSAVC4)
add_subdirectory(test/Conversion/SSAVC4ToVC4)
add_subdirectory(test/CodeGen/SSAVC4)
add_subdirectory(test/CodeGen/VC4Kernel)
