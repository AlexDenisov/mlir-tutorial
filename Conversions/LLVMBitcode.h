#pragma once

#include <llvm/IR/Module.h>
#include <mlir/IR/BuiltinOps.h>

std::unique_ptr<llvm::Module>
covertToBitcode(mlir::MLIRContext &context, llvm::LLVMContext &llvmContext, mlir::ModuleOp module);
