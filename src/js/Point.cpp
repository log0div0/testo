
#include "Point.hpp"
#include "FunctionsAdapters.hpp"
#include <iostream>

namespace js {

static JSClassID class_id = 0;
static JSClassDef class_def = {};

static void finalizer(JSRuntime* rt, JSValue val) {
	nn::Point* point = (nn::Point*)JS_GetOpaque(val, class_id);
	delete point;
}

static const JSCFunctionListEntry proto_funcs[] = {
	Prop("[Symbol.toStringTag]", "Point"),
};

void Point::register_class(ContextRef ctx) {
	if (!class_id) {
		JS_NewClassID(&class_id);
		class_def.class_name = "Point";
		class_def.finalizer = finalizer;
		JS_NewClass(JS_GetRuntime(ctx.handle), class_id, &class_def);
	}

	Value proto = ctx.new_object();
	proto.set_property_function_list(proto_funcs, std::size(proto_funcs));
	ctx.set_class_proto(class_id, std::move(proto));
}

Point::Point(ContextRef ctx, const nn::Point& point): Value(ctx.new_object_class(class_id)) {
	set_opaque(new nn::Point(point));
}

}
