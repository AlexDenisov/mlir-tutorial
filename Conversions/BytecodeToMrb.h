#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <string>

mlir::ModuleOp bytecodeToMrb(mlir::MLIRContext &context, const std::string &filePath);
