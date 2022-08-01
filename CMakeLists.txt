cmake_minimum_required(VERSION 3.22)
project(ruby_mlir_tutorial)

set(CMAKE_CXX_STANDARD 17)

set (MRUBY_DIR ${CMAKE_SOURCE_DIR}/mruby)
include(ExternalProject)
ExternalProject_Add(MRuby
  SOURCE_DIR ${MRUBY_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND rake
  INSTALL_COMMAND ""
  BUILD_IN_SOURCE ON
)

find_package(LLVM CONFIG REQUIRED)
message("Using LLVM_CONFIG: ${LLVM_CONFIG}")
find_package(MLIR CONFIG REQUIRED)
message("Using MLIR_CONFIG: ${MLIR_CONFIG}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

add_subdirectory(Dialects)

add_executable(ruby_mlir_tutorial main.cpp)
add_dependencies(ruby_mlir_tutorial MRuby)
target_link_libraries(ruby_mlir_tutorial PRIVATE
  mruby
  mruby_core
  MLIRStandard
  MRB
)
target_link_directories(ruby_mlir_tutorial PRIVATE ${MRUBY_DIR}/build/host/lib)
target_include_directories(ruby_mlir_tutorial PRIVATE
  ${MRUBY_DIR}/include
  ${CMAKE_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
