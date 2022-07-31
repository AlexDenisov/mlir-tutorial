#include <mruby.h>

int main() {
  auto mrb = mrb_open();
  auto top = mrb_top_self(mrb);
  auto n = mrb_fixnum_value(42);
  mrb_funcall(mrb, top, "puts", 1, n);
  mrb_close(mrb);
  return 0;
}
