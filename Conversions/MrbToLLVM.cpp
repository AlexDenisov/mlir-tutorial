#include "MrbToLLVM.h"

#include "Dialects/MRB/MRB.h"

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Support/DebugAction.h>
#include <mlir/Transforms/DialectConversion.h>

namespace detail {

template <typename Op>
static mlir::Value getOrCreateGlobalString(mlir::OpBuilder &builder, Op op, llvm::StringRef name,
                                           llvm::StringRef value) {
  auto loc = op.getLoc();
  auto nullTerm = value.str() + '\0';
  auto module = op->template getParentOfType<mlir::ModuleOp>();
  // Create the global at the entry of the module.
  mlir::LLVM::GlobalOp global;
  if (!(global = module.template lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(builder.getContext(), 8),
                                               nullTerm.size());
    global = builder.create<mlir::LLVM::GlobalOp>(
        loc, type,
        /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name, builder.getStringAttr(nullTerm),
        /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  mlir::Value cst0 =
      builder.create<mlir::LLVM::ConstantOp>(loc, mlir::IntegerType::get(builder.getContext(), 64),
                                             builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<mlir::LLVM::GEPOp>(
      loc, mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)),
      globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
}

mlir::LLVM::LLVMFuncOp lookupOrCreateFn(mlir::ModuleOp moduleOp, llvm::StringRef name,
                                        llvm::ArrayRef<mlir::Type> paramTypes,
                                        mlir::Type resultType, bool isVarArg = false) {
  auto func = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name);
  if (func)
    return func;
  mlir::OpBuilder b(moduleOp.getBodyRegion());
  return b.create<mlir::LLVM::LLVMFuncOp>(
      moduleOp->getLoc(), name,
      mlir::LLVM::LLVMFunctionType::get(resultType, paramTypes, isVarArg));
}

template <typename Op> struct MrbConversionPattern : public mlir::ConversionPattern {
  MrbConversionPattern(mlir::MLIRContext *context, mlir::TypeConverter &converter)
      : mlir::ConversionPattern(converter, Op::getOperationName(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(mlir::cast<Op>(op), operands, rewriter);
  }
  virtual mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands,
                                              mlir::ConversionPatternRewriter &rewriter) const = 0;
};

} // namespace detail

struct LoadSelfOpLowering : public detail::MrbConversionPattern<mrb::LoadSelfOp> {
  LoadSelfOpLowering(mlir::MLIRContext *ctx, mlir::TypeConverter &converter)
      : MrbConversionPattern(ctx, converter) {}

  mlir::LogicalResult matchAndRewrite(mrb::LoadSelfOp op, llvm::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto state_t = typeConverter->convertType(mrb::stateType::get(getContext()));
    auto value_t = typeConverter->convertType(mrb::valueType::get(getContext()));

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto func = detail::lookupOrCreateFn(module, "rt_load_self", {state_t}, value_t);
    auto replacement = mlir::LLVM::createLLVMCall(rewriter, op->getLoc(), func, operands, value_t);
    rewriter.replaceOp(op, {replacement});
    return mlir::success();
  }
};

struct LoadIOpLowering : public detail::MrbConversionPattern<mrb::LoadIOp> {
  LoadIOpLowering(mlir::MLIRContext *ctx, mlir::TypeConverter &converter)
      : MrbConversionPattern(ctx, converter) {}

  mlir::LogicalResult matchAndRewrite(mrb::LoadIOp op, llvm::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto state_t = typeConverter->convertType(mrb::stateType::get(getContext()));
    auto value_t = typeConverter->convertType(mrb::valueType::get(getContext()));
    auto number_t = op.numberAttr().getType();

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto func = detail::lookupOrCreateFn(module, "rt_load_i", {state_t, number_t}, value_t);
    auto value = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), number_t, op.numberAttr());
    auto replacement = mlir::LLVM::createLLVMCall(rewriter, op->getLoc(), func,
                                                  {operands.front(), value}, value_t);
    rewriter.replaceOp(op, {replacement});
    return mlir::success();
  }
};

struct CallOpLowering : public detail::MrbConversionPattern<mrb::CallOp> {
  CallOpLowering(mlir::MLIRContext *ctx, mlir::TypeConverter &converter)
      : MrbConversionPattern(ctx, converter) {}

  mlir::LogicalResult matchAndRewrite(mrb::CallOp op, llvm::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto state_t = typeConverter->convertType(mrb::stateType::get(getContext()));
    auto value_t = typeConverter->convertType(mrb::valueType::get(getContext()));

    auto state = operands[0];
    auto receiver = operands[1];
    auto calleeName = detail::getOrCreateGlobalString(rewriter, op, op.name(), op.name());
    auto argc = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), op.argcAttr().getType(),
                                                        op.argcAttr());

    llvm::SmallVector<mlir::Value> argv({state, receiver, calleeName, argc});
    for (size_t i = 2; i < operands.size(); i++) {
      // push all the variadic operands
      argv.push_back(operands[i]);
    }

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto func = detail::lookupOrCreateFn(
        module, "mrb_funcall", {state_t, value_t, calleeName.getType(), op.argcAttr().getType()},
        value_t,
        /* isVarArg= */ true);

    auto replacement = mlir::LLVM::createLLVMCall(rewriter, op->getLoc(), func, argv, value_t);
    rewriter.replaceOp(op, {replacement});
    return mlir::success();
  }
};

void mrbToLLVM(mlir::MLIRContext &context, mlir::ModuleOp module) {
  mlir::ConversionTarget target(context);
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&context);
  typeConverter.addConversion([&](mrb::valueType type) -> llvm::Optional<mlir::Type> {
    auto converted = mlir::LLVM::LLVMStructType::getIdentified(&context, "mrb_value");
    if (converted.setBody({mlir::IntegerType::get(&context, 64)}, false).failed()) {
      assert(false && "Cannot set body of the mrb_value struct twice");
    }
    return converted;
  });
  typeConverter.addConversion([&](mrb::stateType type) -> llvm::Optional<mlir::Type> {
    auto llvmType = mlir::LLVM::LLVMStructType::getOpaque("mrb_state", &context);
    return mlir::LLVM::LLVMPointerType::get(llvmType);
  });

  mlir::RewritePatternSet patterns(&context);
  patterns.add<LoadIOpLowering, CallOpLowering, LoadSelfOpLowering>(&context, typeConverter);

  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (mlir::failed(mlir::applyFullConversion(module.getOperation(), target, frozenPatterns))) {
    module.print(llvm::errs());
    llvm::errs() << "Cannot apply LLVM conversion\n";
    abort();
  }
}
