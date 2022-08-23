#include <mruby.h>

mrb_value top(mrb_state *mrb, mrb_value self) {
  mrb_value v = mrb_fixnum_value(42);
  return mrb_funcall(mrb, self, "puts", 1, v);
}

int main() {
  mrb_state *mrb = mrb_open();
  mrb_value self = mrb_top_self(mrb);
  top(mrb, self);
  mrb_close(mrb);
  return 0;
}
