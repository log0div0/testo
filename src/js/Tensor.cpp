
#include "Tensor.hpp"
#include "FunctionsAdapters.hpp"
#include <iostream>

namespace js {

static JSClassID class_id = 0;
static JSClassDef class_def = {};

static void finalizer(JSRuntime* rt, JSValue val) {
	nn::Tensor* tensor = (nn::Tensor*)JS_GetOpaque(val, class_id);
	delete tensor;
}

static Value size(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	return ctx.new_int32(tensor->size());
}

static Value match(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::match");
	}
	std::string text = args.at(0);
	return Tensor(ctx, tensor->match(text));
}

static Value match_foreground(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::match_foreground");
	}
	std::string color = args.at(0);
	return Tensor(ctx, tensor->match_foreground(ctx.image(), color));
}

static Value match_background(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::match_background");
	}
	std::string color = args.at(0);
	return Tensor(ctx, tensor->match_background(ctx.image(), color));
}

static const JSCFunctionListEntry proto_funcs[] = {
	Method<size>("size", 0),
	Method<match>("match", 0),
	Method<match_foreground>("match_foreground", 0),
	Method<match_background>("match_background", 0),
	Prop("[Symbol.toStringTag]", "Tensor"),
};

void Tensor::register_class(ContextRef ctx) {
	if (!class_id) {
		JS_NewClassID(&class_id);
		class_def.class_name = "Tensor";
		class_def.finalizer = finalizer;
		JS_NewClass(JS_GetRuntime(ctx.handle), class_id, &class_def);
	}

	Value proto = ctx.new_object();
	proto.set_property_function_list(proto_funcs, std::size(proto_funcs));
	ctx.set_class_proto(class_id, std::move(proto));
}

Tensor::Tensor(ContextRef ctx, const nn::Tensor& tensor): Value(ctx.new_object_class(class_id)) {
	set_opaque(new nn::Tensor(tensor));
}

}
