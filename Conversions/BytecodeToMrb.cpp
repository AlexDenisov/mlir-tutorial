#include "BytecodeToMrb.h"

#include "Dialects/MRB/MRB.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <mruby.h>
#include <mruby/compile.h>
#include <mruby/opcode.h>
#include <mruby/proc.h>

extern "C" {
const char *mrb_debug_get_filename(mrb_state *mrb, const mrb_irep *irep, uint32_t pc);
int32_t mrb_debug_get_line(mrb_state *mrb, const mrb_irep *irep, uint32_t pc);
void mrb_codedump_all(mrb_state *mrb, struct RProc *proc);
}

const char *opcode_name(mrb_code code) {
#include "mruby/opcode.h"
  switch (code) {
#define OPCODE(x, _)                                                                               \
  case OP_##x:                                                                                     \
    return "OP_" #x;
#include "mruby/ops.h"
#undef OPCODE
  }
  return "unknown opcode?";
}

struct Regs {
  uint32_t a;
  uint32_t b;
  uint32_t c;
};

mlir::ModuleOp procToModule(mlir::MLIRContext &context, struct mrb_state *mrb, struct RProc *proc) {
  assert(!MRB_PROC_CFUNC_P(proc));

  const char *filename = mrb_debug_get_filename(mrb, proc->body.irep, 0);
  auto functionLocation = mlir::FileLineColLoc::get(&context, filename, 0, 0);
  auto module = mlir::ModuleOp::create(functionLocation, llvm::StringRef(filename));
  mlir::Type mrb_state_t(mrb::stateType::get(&context));
  mlir::Type mrb_value_t(mrb::valueType::get(&context));

  mlir::OpBuilder builder(&context);

  auto functionType = builder.getFunctionType({mrb_state_t, mrb_value_t}, {mrb_value_t});
  auto function = mlir::FuncOp::create(functionLocation, "top", functionType);
  builder.setInsertionPointToStart(function.addEntryBlock());

  auto state = function.getArgument(0);

  const mrb_irep *irep = proc->body.irep;
  for (uint16_t pc_offset = 0; pc_offset < irep->ilen; pc_offset++) {
    const mrb_code *pc_base = (irep->iseq + pc_offset);
    const mrb_code *pc = pc_base;
    int32_t line = mrb_debug_get_line(mrb, irep, pc - irep->iseq);
    auto location = mlir::FileLineColLoc::get(&context, filename, line, pc_offset);

    Regs regs{};
    auto opcode = (mrb_insn)*pc;
    pc++;
    switch (opcode) {
    case OP_NOP:
    case OP_STOP:
      // NOOP
      break;

    case OP_LOADI: {
      // OPCODE(LOADI,      BB)       /* R(a) = mrb_int(b) */
      regs.a = READ_B();
      regs.b = READ_B();
      builder.create<mrb::LoadIOp>(location, mrb_value_t, state,
                                   builder.getUI32IntegerAttr(regs.b));
    } break;

    case OP_LOADSELF: {
      // OPCODE(LOADSELF,   B)        /* R[a] = self */
      regs.a = READ_B();
      builder.create<mrb::LoadSelfOp>(location, mrb_value_t, state);
    } break;

    case OP_SSEND: {
      // clang-format off
      // OPCODE(SSEND,      BBB)      /* R[a] = self.send(Syms[b],R[a+1]..,R[a+n+1]:R[a+n+2]..)(c=n|k<<4) */
      // clang-format on
      regs.a = READ_B();
      regs.b = READ_B();
      regs.c = READ_B();

      auto argc = regs.c;
      auto callName = mrb_sym_name(mrb, irep->syms[regs.b]);
      auto receiver = builder.create<mrb::LoadSelfOp>(location, mrb_value_t, state);
      llvm::SmallVector<mlir::Value> argv;
      for (size_t i = 0; i < argc; i++) {
        auto arg = builder.create<mrb::LoadIOp>(location, mrb_value_t, state,
                                                builder.getUI32IntegerAttr(42));
        argv.push_back(arg);
      }
      builder.create<mrb::CallOp>(location, mrb_value_t, state, receiver,
                                  builder.getStringAttr(callName), builder.getUI32IntegerAttr(argc),
                                  argv);
    } break;
    case OP_RETURN: {
      // OPCODE(RETURN,     B)        /* return R[a] (normal) */
      regs.a = READ_B();
      auto retVal = builder.create<mrb::LoadIOp>(location, mrb_value_t, state,
                                              builder.getUI32IntegerAttr(42));
      builder.create<mlir::ReturnOp>(location, mlir::TypeRange({mrb_value_t}), mlir::ValueRange({retVal}));
    } break;
    default: {
      llvm::errs() << "Unsupported op: " << opcode_name(opcode) << "\n";
      abort();
    }
    }
    pc_offset += pc - pc_base - 1;
  }

  module.push_back(function);
  return module;
}

mlir::ModuleOp bytecodeToMrb(mlir::MLIRContext &context, const std::string &filePath) {
  /// TODO: error handling
  /// TODO: resource deallocation
  FILE *input = fopen(filePath.c_str(), "r");
  assert(input && "Cannot open input file?");
  mrb_state *state = mrb_open();
  mrbc_context *c = mrbc_context_new(state);

  mrbc_filename(state, c, filePath.c_str());
  struct mrb_parser_state *rubyAST = mrb_parse_file(state, input, c);
  struct RProc *proc = mrb_generate_code(state, rubyAST);
  mrb_codedump_all(state, proc);
  return procToModule(context, state, proc);
}
