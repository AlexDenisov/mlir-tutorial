#include <llvm/Support/ManagedStatic.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>

int main() {
  llvm::llvm_shutdown_obj shutdownGuard;

  mlir::DialectRegistry registry;
  registry.insert<mlir::StandardOpsDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OpBuilder builder(&context);
  auto location = mlir::FileLineColLoc::get(builder.getStringAttr("<memory>"), 0, 0);

  auto module = builder.create<mlir::ModuleOp>(location);
  auto functionType = builder.getFunctionType({}, builder.getIntegerType(64));
  auto function = builder.create<mlir::FuncOp>(location, "top", functionType);
  module.push_back(function);
  builder.setInsertionPointToStart(function.addEntryBlock());

  auto x = builder.create<mlir::ConstantOp>(location,
                                            builder.getIntegerAttr(builder.getIntegerType(64), 42));
  builder.create<mlir::ReturnOp>(location, mlir::ValueRange({x}));

  module.print(llvm::errs());

  return 0;
}
