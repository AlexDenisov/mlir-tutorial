#include "Dialects/MRB/MRB.h"

#include "Conversions/LLVMBitcode.h"
#include "Conversions/MrbToLLVM.h"

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ManagedStatic.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>

llvm::cl::OptionCategory FireStormCategory("mlir-tutorial");

// llvm::cl::opt<std::string> Input(llvm::cl::Positional, llvm::cl::Required,
//                                  llvm::cl::desc("Input file"), llvm::cl::cat(FireStormCategory));

llvm::cl::opt<std::string> Output("output", llvm::cl::desc("Output object file"),
                                  llvm::cl::cat(FireStormCategory));

void saveBitcode(std::unique_ptr<llvm::Module> &module) {
  std::error_code error;
  llvm::raw_fd_ostream stream(Output.getValue(), error);
  if (error) {
    llvm::errs() << "Cannot create output file: " << Output.getValue() << ": " << error.message()
                 << "\n";
    abort();
  }

  llvm::WriteBitcodeToFile(*module, stream);
}

int main(int argc, char **argv) {
  llvm::llvm_shutdown_obj shutdownGuard;

  llvm::cl::HideUnrelatedOptions(FireStormCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::DialectRegistry registry;
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mrb::MrbDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OpBuilder builder(&context);
  auto location = mlir::FileLineColLoc::get(builder.getStringAttr("<memory>"), 0, 0);

  auto module = builder.create<mlir::ModuleOp>(location);

  auto state_t = mrb::stateType::get(&context);
  auto value_t = mrb::valueType::get(&context);

  auto functionType = builder.getFunctionType({state_t, value_t}, value_t);
  auto function = builder.create<mlir::FuncOp>(location, "top", functionType);
  module.push_back(function);
  builder.setInsertionPointToStart(function.addEntryBlock());

  auto self = builder.create<mrb::LoadSelfOp>(location, value_t, function.getArgument(0));
  auto number = builder.create<mrb::LoadIOp>(location, value_t, function.getArgument(0),
                                             builder.getUI32IntegerAttr(42));
  auto call = builder.create<mrb::CallOp>(location, value_t, function.getArgument(0), self,
                                          builder.getStringAttr("puts"),
                                          builder.getUI32IntegerAttr(1), mlir::ValueRange{number});
  builder.create<mlir::ReturnOp>(location, mlir::ValueRange({call}));

  module.print(llvm::errs());

  mrbToLLVM(context, module);

  module.print(llvm::errs());

  llvm::LLVMContext llvmContext;
  auto bitcode = covertToBitcode(context, llvmContext, module);
  bitcode->print(llvm::errs(), nullptr);

  saveBitcode(bitcode);

  return 0;
}
