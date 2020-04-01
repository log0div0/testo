
#pragma once

#include "Context.hpp"
#include <nn/Rect.hpp>

namespace js {

struct Rect: Value {
	static void register_class(ContextRef ctx);

	Rect(ContextRef ctx, const nn::Rect& rect);
};

}
