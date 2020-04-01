
#include "Rect.hpp"
#include "FunctionsAdapters.hpp"
#include <iostream>

namespace js {

static JSClassID class_id = 0;
static JSClassDef class_def = {};

static void finalizer(JSRuntime* rt, JSValue val) {
	nn::Rect* rect = (nn::Rect*)JS_GetOpaque(val, class_id);
	delete rect;
}

static Value toJSON(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Rect* rect = (nn::Rect*)this_val.get_opaque(class_id);
	Value result = ctx.new_object();
	result.set_property("left", ctx.new_int32(rect->left));
	result.set_property("right", ctx.new_int32(rect->right));
	result.set_property("top", ctx.new_int32(rect->top));
	result.set_property("bottom", ctx.new_int32(rect->bottom));
	return result;
}

static Value x(ContextRef ctx, ValueRef this_val) {
	nn::Rect* rect = (nn::Rect*)this_val.get_opaque(class_id);
	return ctx.new_int32(rect->center_x());
}

static Value y(ContextRef ctx, ValueRef this_val) {
	nn::Rect* rect = (nn::Rect*)this_val.get_opaque(class_id);
	return ctx.new_int32(rect->center_y());
}

static const JSCFunctionListEntry proto_funcs[] = {
	Method<toJSON>("toJSON", 0),
	GetSet<x, nullptr>("x"),
	GetSet<y, nullptr>("y"),
	Prop("[Symbol.toStringTag]", "Rect"),
};

void Rect::register_class(ContextRef ctx) {
	if (!class_id) {
		JS_NewClassID(&class_id);
		class_def.class_name = "Rect";
		class_def.finalizer = finalizer;
		JS_NewClass(JS_GetRuntime(ctx.handle), class_id, &class_def);
	}

	Value proto = ctx.new_object();
	proto.set_property_function_list(proto_funcs, std::size(proto_funcs));
	ctx.set_class_proto(class_id, std::move(proto));
}

Rect::Rect(ContextRef ctx, const nn::Rect& rect): Value(ctx.new_object_class(class_id)) {
	set_opaque(new nn::Rect(rect));
}

}
