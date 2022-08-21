add_library(Conversions
  MrbToLLVM.h
  MrbToLLVM.cpp
  LLVMBitcode.cpp LLVMBitcode.h BytecodeToMrb.cpp BytecodeToMrb.h)

target_link_libraries(Conversions PRIVATE
  MLIRStandard
  MLIRLLVMIR
  MLIRLLVMCommonConversion
  MLIRStandardToLLVM
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  MRB
  mruby_core
  mruby
)
target_include_directories(Conversions PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${MRUBY_DIR}/include
)
target_link_directories(Conversions PRIVATE ${MRUBY_DIR}/build/host/lib)
