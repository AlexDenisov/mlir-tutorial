#include <mruby.h>

extern mrb_value top(mrb_state *, mrb_value);

mrb_value rt_load_i(mrb_state *mrb, uint32_t val) { return mrb_fixnum_value(val); }

mrb_value rt_load_self(mrb_state *mrb) { return mrb->c->ci->stack[0]; }

int main() {
  mrb_state *mrb = mrb_open();
  mrb_value self = mrb_top_self(mrb);
  top(mrb, self);
  mrb_close(mrb);

  return 0;
}
