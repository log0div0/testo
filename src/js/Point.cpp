
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

static Value toJSON(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	Value result = ctx.new_object();
	result.set_property_str("x", ctx.new_int32(point->x));
	result.set_property_str("y", ctx.new_int32(point->y));
	return result;
}

static Value x(ContextRef ctx, ValueRef this_val) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	return ctx.new_int32(point->x);
}

static Value x(ContextRef ctx, ValueRef this_val, ValueRef val) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	return ctx.new_int32(point->x = val);
}

static Value y(ContextRef ctx, ValueRef this_val) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	return ctx.new_int32(point->y);
}

static Value y(ContextRef ctx, ValueRef this_val, ValueRef val) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	return ctx.new_int32(point->y = val);
}

static Value move_up(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Point::move_up");
	}
	int32_t offset = args.at(0);
	return Point(ctx, point->move_up(offset));
}

static Value move_down(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Point::move_down");
	}
	int32_t offset = args.at(0);
	return Point(ctx, point->move_down(offset));
}

static Value move_left(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Point::move_left");
	}
	int32_t offset = args.at(0);
	return Point(ctx, point->move_left(offset));
}

static Value move_right(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Point* point = (nn::Point*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Point::move_right");
	}
	int32_t offset = args.at(0);
	return Point(ctx, point->move_right(offset));
}

static const JSCFunctionListEntry proto_funcs[] = {
	Method<toJSON>("toJSON"),
	GetSet<x, x>("x"),
	GetSet<y, y>("y"),
	Method<move_up>("move_up"),
	Method<move_down>("move_down"),
	Method<move_left>("move_left"),
	Method<move_right>("move_right"),
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
