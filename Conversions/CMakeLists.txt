add_library(Conversions
  MrbToLLVM.h
  MrbToLLVM.cpp
)
target_link_libraries(Conversions PRIVATE
  MLIRStandard
  MLIRLLVMIR
  MLIRLLVMCommonConversion
  MLIRStandardToLLVM
  MRB
)
target_include_directories(Conversions PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
