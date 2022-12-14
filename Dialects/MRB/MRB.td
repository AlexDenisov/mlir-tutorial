include "mlir/IR/OpBase.td"

def Mrb_Dialect : Dialect {
  let name = "mrb";
  let summary = "A high-level dialect for MRuby";
  let cppNamespace = "mrb";
}

class MrbType<string name> : TypeDef<Mrb_Dialect, name> {
  let summary = name;
  let mnemonic = name;
}

def StateType : MrbType<"state"> {}
def ValueType : MrbType<"value"> {}

class Mrb_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mrb_Dialect, mnemonic, traits>;

def LoadSelfOp : Mrb_Op<"load_self"> {
  let summary = "Loads self";
  let arguments = (ins StateType:$state);
  let results = (outs ValueType);
}

def LoadIOp : Mrb_Op<"load_i"> {
  let summary = "Loads an integer";
  let arguments = (ins StateType:$state, UI32Attr:$number);
  let results = (outs ValueType);
}

def CallOp : Mrb_Op<"call"> {
  let summary = "Call a function (send message)";
  let arguments = (ins StateType:$state, ValueType:$receiver, StrAttr:$name, UI32Attr:$argc, Variadic<ValueType>:$argv);
  let results = (outs ValueType);
}
