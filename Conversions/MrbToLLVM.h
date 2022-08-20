#pragma once

#include <mlir/IR/BuiltinOps.h>

void mrbToLLVM(mlir::MLIRContext &context, mlir::ModuleOp module);
