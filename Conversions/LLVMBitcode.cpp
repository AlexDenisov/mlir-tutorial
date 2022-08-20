#include "LLVMBitcode.h"

#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

std::unique_ptr<llvm::Module>
covertToBitcode(mlir::MLIRContext &context, llvm::LLVMContext &llvmContext, mlir::ModuleOp module) {
  mlir::registerLLVMDialectTranslation(context);
  auto ir = mlir::translateModuleToLLVMIR(module, llvmContext, "MRubyModule");
  if (!ir) {
    llvm::errs() << "Cannot translate LLVM\n";
    abort();
  }
  return std::move(ir);
}
