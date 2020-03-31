
#include "Context.hpp"
#include <vector>

namespace js {

Value js_print(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args);
Value detect_text(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args);

}
