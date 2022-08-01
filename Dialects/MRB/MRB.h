#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "MRBDialect.h.inc"
#include "MRBEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "MRBTypeDefs.h.inc"

#define GET_OP_CLASSES
#include "MRBOps.h.inc"
