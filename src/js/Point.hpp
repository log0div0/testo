
#pragma once

#include "Context.hpp"
#include <nn/OCR.hpp>

namespace js {

struct Point: Value {
	static void register_class(ContextRef ctx);

	Point(ContextRef ctx, const nn::Point& point);
};

}
