# Quarantine the optional Triton C++ parser/link closure behind one interface
# target. This file is included only when VC4_ENABLE_TRITON_CPP_FRONTEND=ON.

if(NOT VC4_ENABLE_TRITON_CPP_FRONTEND)
  message(FATAL_ERROR
    "VC4TritonFrontendDeps.cmake must only be included when "
    "VC4_ENABLE_TRITON_CPP_FRONTEND=ON")
endif()

if(TARGET VC4TritonFrontendDeps)
  return()
endif()

set(VC4_TRITON_EXPECTED_LLVM_HASH
  "ac5dc54d509169d387fcfd495d71853d81c46484")

foreach(var
    VC4_TRITON_SOURCE_DIR
    VC4_TRITON_BUILD_DIR
    VC4_TRITON_LLVM_PREFIX)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR
      "${var} is required when VC4_ENABLE_TRITON_CPP_FRONTEND=ON")
  endif()
endforeach()

set(VC4_TRITON_LLVM_HASH_FILE
  "${VC4_TRITON_SOURCE_DIR}/cmake/llvm-hash.txt")
set(VC4_TRITON_DIALECT_HEADER
  "${VC4_TRITON_SOURCE_DIR}/include/triton/Dialect/Triton/IR/Dialect.h")
set(VC4_TRITON_GENERATED_INCLUDE_DIR
  "${VC4_TRITON_BUILD_DIR}/include")
set(VC4_TRITON_LLVM_CONFIG
  "${VC4_TRITON_LLVM_PREFIX}/lib/cmake/llvm/LLVMConfig.cmake")
set(VC4_TRITON_MLIR_CONFIG
  "${VC4_TRITON_LLVM_PREFIX}/lib/cmake/mlir/MLIRConfig.cmake")

foreach(path
    "${VC4_TRITON_LLVM_HASH_FILE}"
    "${VC4_TRITON_DIALECT_HEADER}"
    "${VC4_TRITON_GENERATED_INCLUDE_DIR}"
    "${VC4_TRITON_LLVM_CONFIG}"
    "${VC4_TRITON_MLIR_CONFIG}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR
      "Required Triton frontend dependency path does not exist: ${path}")
  endif()
endforeach()

file(READ "${VC4_TRITON_LLVM_HASH_FILE}" VC4_TRITON_ACTUAL_LLVM_HASH)
string(STRIP "${VC4_TRITON_ACTUAL_LLVM_HASH}" VC4_TRITON_ACTUAL_LLVM_HASH)
if(NOT VC4_TRITON_ACTUAL_LLVM_HASH STREQUAL VC4_TRITON_EXPECTED_LLVM_HASH)
  message(FATAL_ERROR
    "Triton llvm-hash.txt mismatch: expected "
    "${VC4_TRITON_EXPECTED_LLVM_HASH}, got ${VC4_TRITON_ACTUAL_LLVM_HASH}")
endif()

set(VC4_TRITON_FRONTEND_OBJECT_COUNT 0)
foreach(obj IN LISTS VC4_TRITON_CPP_OBJECTS)
  if(NOT IS_ABSOLUTE "${obj}")
    message(FATAL_ERROR
      "VC4_TRITON_CPP_OBJECTS entries must be explicit absolute paths: ${obj}")
  endif()
  if(NOT EXISTS "${obj}")
    message(FATAL_ERROR
      "VC4_TRITON_CPP_OBJECTS entry does not exist: ${obj}")
  endif()
  math(EXPR VC4_TRITON_FRONTEND_OBJECT_COUNT
    "${VC4_TRITON_FRONTEND_OBJECT_COUNT} + 1")
endforeach()

set(VC4_TRITON_FRONTEND_LIBRARY_COUNT 0)
foreach(lib IN LISTS VC4_TRITON_CPP_LIBRARIES)
  if(IS_ABSOLUTE "${lib}" AND NOT EXISTS "${lib}")
    message(FATAL_ERROR
      "VC4_TRITON_CPP_LIBRARIES absolute entry does not exist: ${lib}")
  endif()
  math(EXPR VC4_TRITON_FRONTEND_LIBRARY_COUNT
    "${VC4_TRITON_FRONTEND_LIBRARY_COUNT} + 1")
endforeach()

if(VC4_TRITON_FRONTEND_OBJECT_COUNT EQUAL 0 AND
    VC4_TRITON_FRONTEND_LIBRARY_COUNT EQUAL 0)
  message(FATAL_ERROR
    "VC4_TRITON_CPP_OBJECTS or VC4_TRITON_CPP_LIBRARIES must provide the "
    "explicit Triton C++ dependency closure")
endif()

add_library(VC4TritonFrontendDeps INTERFACE)
target_include_directories(VC4TritonFrontendDeps INTERFACE
  "${VC4_TRITON_SOURCE_DIR}/include"
  "${VC4_TRITON_GENERATED_INCLUDE_DIR}"
)
target_link_libraries(VC4TritonFrontendDeps INTERFACE
  ${VC4_TRITON_CPP_OBJECTS}
  ${VC4_TRITON_CPP_LIBRARIES}
)

message(STATUS "VC4 Triton frontend source dir: ${VC4_TRITON_SOURCE_DIR}")
message(STATUS "VC4 Triton frontend build dir: ${VC4_TRITON_BUILD_DIR}")
message(STATUS "VC4 Triton frontend LLVM prefix: ${VC4_TRITON_LLVM_PREFIX}")
message(STATUS
  "VC4 Triton frontend explicit object files: "
  "${VC4_TRITON_FRONTEND_OBJECT_COUNT}")
message(STATUS
  "VC4 Triton frontend explicit libraries: "
  "${VC4_TRITON_FRONTEND_LIBRARY_COUNT}")
