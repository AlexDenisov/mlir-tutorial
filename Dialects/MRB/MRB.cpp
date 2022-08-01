#include "MRB.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mrb;

#include "MRBDialect.cpp.inc"
#include "MRBEnums.cpp.inc"

void MrbDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MRBOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "MRBTypeDefs.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "MRBOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "MRBTypeDefs.cpp.inc"
