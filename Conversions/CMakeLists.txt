add_library(Conversions
  MrbToLLVM.h
  MrbToLLVM.cpp
  LLVMBitcode.cpp LLVMBitcode.h)
target_link_libraries(Conversions PRIVATE
  MLIRStandard
  MLIRLLVMIR
  MLIRLLVMCommonConversion
  MLIRStandardToLLVM
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MRB
)
target_include_directories(Conversions PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)
