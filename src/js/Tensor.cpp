
#include "Tensor.hpp"
#include "Point.hpp"
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

static Value from_top(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::from_top");
	}
	int32_t index = args.at(0);
	return Tensor(ctx, tensor->from_top(index));
}

static Value from_bottom(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::from_bottom");
	}
	int32_t index = args.at(0);
	return Tensor(ctx, tensor->from_bottom(index));
}

static Value from_left(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::from_left");
	}
	int32_t index = args.at(0);
	return Tensor(ctx, tensor->from_left(index));
}

static Value from_right(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::from_right");
	}
	int32_t index = args.at(0);
	return Tensor(ctx, tensor->from_right(index));
}

static Value x(ContextRef ctx, ValueRef this_val) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	return ctx.new_int32(tensor->x());
}

static Value y(ContextRef ctx, ValueRef this_val) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	return ctx.new_int32(tensor->y());
}

static Value left_top(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::left_top");
	}
	return Point(ctx, tensor->left_top());
}

static Value left_bottom(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::left_bottom");
	}
	return Point(ctx, tensor->left_bottom());
}

static Value right_top(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::right_top");
	}
	return Point(ctx, tensor->right_top());
}

static Value right_bottom(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::right_bottom");
	}
	return Point(ctx, tensor->right_bottom());
}

static Value center(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::center");
	}
	return Point(ctx, tensor->center());
}

static Value center_top(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::center_top");
	}
	return Point(ctx, tensor->center_top());
}

static Value center_bottom(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::center_bottom");
	}
	return Point(ctx, tensor->center_bottom());
}

static Value left_center(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::left_center");
	}
	return Point(ctx, tensor->left_center());
}

static Value right_center(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	nn::Tensor* tensor = (nn::Tensor*)this_val.get_opaque(class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::right_center");
	}
	return Point(ctx, tensor->right_center());
}

static const JSCFunctionListEntry proto_funcs[] = {
	Method<size>("size"),
	Method<match>("match"),
	Method<match_foreground>("match_foreground"),
	Method<match_background>("match_background"),
	Method<from_top>("from_top"),
	Method<from_bottom>("from_bottom"),
	Method<from_left>("from_left"),
	Method<from_right>("from_right"),
	GetSet<x, nullptr>("x"),
	GetSet<y, nullptr>("y"),
	Method<left_top>("left_top"),
	Method<left_bottom>("left_bottom"),
	Method<right_top>("right_top"),
	Method<right_bottom>("right_bottom"),
	Method<center>("center"),
	Method<center_top>("center_top"),
	Method<center_bottom>("center_bottom"),
	Method<left_center>("left_center"),
	Method<right_center>("right_center"),
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
