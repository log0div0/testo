
#pragma once

#include "Context.hpp"
#include <nn/Point.hpp>

namespace js {

struct Point: Value {
	static JSClassID class_id;
	static JSClassDef class_def;

	static void register_class(ContextRef ctx);

	Point(ContextRef ctx, const nn::Point& point);
};

}
