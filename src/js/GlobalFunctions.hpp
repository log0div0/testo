
#pragma once

#include "Context.hpp"
#include <vector>

namespace js {

Value print(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args);
Value find_text(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args);

}
